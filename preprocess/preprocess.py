import sys
import argparse
import numpy as np
import h5py
from collections import defaultdict
from transformers import *
import json
from tqdm import tqdm
from . import extract
from util.util import Indexer, load_arg_role, parse_label, is_any_vnclass
from hf.v_model import VModel

def extract_semlink(sense_vn_map, suffix_vn_map):
	rs = {}
	for suffix, mapping in suffix_vn_map.items():
		for vc_id, pairs in mapping.items():
			if not pairs:
				continue
			if (vc_id, suffix) not in rs:
				rs[(vc_id, suffix)] = []
			rs[(vc_id, suffix)].extend(pairs)
	return rs


def load_semlink(semlink_path):
	sense_vn_map, suffix_vn_map = {}, {}
	with open(semlink_path + '.sense_vn.json', 'r') as f:
		sense_vn_map = json.load(f)
	with open(semlink_path + '.suffix_vn.json', 'r') as f:
		suffix_vn_map = json.load(f)
	return sense_vn_map, suffix_vn_map


def manual_fix_for_semlink(semlink, collapse_vc):
	semlink[('47.8', '01')].append(('ARG2', 'CO_THEME'))
	semlink[('47.8', '01')].append(('ARG1', 'CO_THEME'))
	semlink[('36.2', '01')].append(('ARG1', 'CO_AGENT'))
	semlink[('90', '01')].append(('ARG1', 'CO_THEME'))
	semlink[('22.4', '01')].append(('ARG2', 'CO_PATIENT'))
	semlink[('13.6.2', '01')].append(('ARG1', 'CO_THEME'))
	semlink[('36.2', '01')].append(('ARG2', 'CO_AGENT'))
	if not collapse_vc:
		semlink[('22.1-1-1', '01')].append(('ARG2', 'CO_PATIENT'))
		semlink[('36.3-1', '01')].append(('ARG1', 'CO_AGENT'))
		semlink[('36.1.1', '01')].append(('ARG1', 'CO_AGENT'))
		semlink[('22.2-2', '01')].append(('ARG2', 'CO_PATIENT'))
	return semlink


def get_tokenizer(key):
	model_map={"bert-base-uncased": (BertModel, BertTokenizer),
		"roberta-base": (RobertaModel, RobertaTokenizer)}
	model_cls, tokenizer_cls = model_map[key]
	print('loading tokenizer: {0}'.format(key))
	tokenizer = tokenizer_cls.from_pretrained(key)
	return tokenizer

			
def pad(ls, length, symbol, pad_back = True):
	if len(ls) >= length:
		return ls[:length]
	if pad_back:
		return ls + [symbol] * (length -len(ls))
	else:
		return [symbol] * (length -len(ls)) + ls		


def make_label_vocab2(opt, label_indexer, labels, count):
	for _, label_orig in enumerate(labels):
		label_orig = label_orig.strip().split()
		label_indexer.register_all_words(label_orig, count)
	return len(labels)


def make_label_vocab1(opt, label_indexer, labels, count):
	for _, label_orig in enumerate(labels):
		label_indexer.register_all_words([label_orig], count)
	return len(labels)


def load_frameset(path):
	rs = {}
	with open(path, 'r') as f:
		for line in f:
			parts = line.strip().split(' ')
			lemma = parts[0]
			rs[lemma] = []
			if lemma == '#':
				continue
			for p in parts[1:]:
				roleset_id, arg_set = p.split('|')
				rs[lemma].append((roleset_id, [p for p in arg_set.split(',')]))
	return rs


def get_arg_mask(label_dict, args):
	mask = np.zeros((len(label_dict),))
	for arg in args:
		for l, idx in label_dict.items():
			if arg in l:
				mask[idx] = 1.0

	# always enable modifier and V
	for l, idx in label_dict.items():
		if 'M-' in l or '-V' in l:
			mask[idx] = 1.0

	# always enable *A role
	for l, idx in label_dict.items():
		if 'ARGA' in l or 'AA' in l:
			mask[idx] = 1.0
	return mask


# return frame indexer (by lemma)
#	and frame_pool (i.e. vectorized frameset)
def extract_frame_pool(suffix_indexer, srl_label_indexer, frame_indexer, frameset):
	frame_pool = np.zeros((len(frame_indexer.d), len(suffix_indexer.d), len(srl_label_indexer.d)))
	for i, (lemma, frames) in enumerate(frameset.items()):
		# by default, the first frame is for those mismatches
		if i == 0:
			frame_pool[i, :, :] = 1.0
			continue

		# setup role sets one by one
		for frame_id, args in frames:
			frame_idx = suffix_indexer.d[frame_id]
			frame_pool[i, frame_idx] = get_arg_mask(srl_label_indexer.d, args)

		# enable O for any frame_id, even if that one is not applicable
		frame_pool[i, :, 0] = 1.0

	print('frame_pool shape:', frame_pool.shape)
	return frame_pool


def convert(opt, tokenizer, vn_label_indexer, vn_class_indexer, srl_label_indexer, roleset_indexer, roleset_suffix_indexer, frame_indexer, frame_pool,
		toks, tok_l, vn_labels, vn_classes, srl_labels, roleset_ids, roleset_suffixes, orig_toks, v_ids,
		semlink_map, output):
	np.random.seed(opt.seed)

	act_max_wp_l = 0
	act_max_wpa_l = 0

	grouped_input = group_tokenized_with_labels(opt, toks, tok_l, vn_labels, vn_classes, srl_labels, roleset_ids, roleset_suffixes, orig_toks, v_ids, output)
	num_ex = len(grouped_input)

	tok_idx = np.zeros((num_ex, opt.max_seq_l), dtype=np.int32)
	wp_idx = np.zeros((num_ex, opt.max_wp_l), dtype=np.int32) + tokenizer.pad_token_id	# make it padding which will be auto filtered by transformer.
	wpa_idx = np.zeros((num_ex, opt.max_wpa_l), dtype=np.int32) + tokenizer.pad_token_id	# make it padding which will be auto filtered by transformer.
	sub2tok_idx = np.zeros((num_ex, opt.max_seq_l, opt.max_num_subtok), dtype=np.int32) - 1	# empty is -1
	v_idx = np.zeros((num_ex, opt.max_num_v), dtype=np.int32)
	vn_label = np.zeros((num_ex, opt.max_num_v, opt.max_seq_l), dtype=np.int32)
	# TODO, NOT USED
	vn_class = np.zeros((num_ex, opt.max_num_v), dtype=np.int32)
	srl_label = np.zeros((num_ex, opt.max_num_v, opt.max_seq_l), dtype=np.int32)
	#
	frame_idx = np.zeros((num_ex, opt.max_num_v), dtype=int)
	#
	v_length = np.zeros((num_ex,), dtype=np.int32)	# number of v
	roleset_id = np.zeros((num_ex, opt.max_num_v), dtype=np.int32)	# just output, but deprecated
	roleset_suffix = np.zeros((num_ex, opt.max_num_v), dtype=np.int32)
	seq_length = np.zeros((num_ex,), dtype=np.int32)
	orig_seq_length = np.zeros((num_ex,), dtype=np.int32)
	ex_idx = np.zeros(num_ex, dtype=np.int32)
	semlink = np.zeros((num_ex, opt.max_num_v, 2, opt.max_num_semlink), dtype=np.int32) - 1	# empty is -1
	semlink_l = np.zeros((num_ex, opt.max_num_v), dtype=np.int32)
	batch_keys = np.array([None for _ in range(num_ex)])

	ex_id = 0
	num_prop = 0
	semlink_cnt = 0
	for _, (cur_toks, cur_tok_l, cur_vn_labels, cur_vn_classes, cur_srl_labels, cur_roleset_ids, cur_roleset_suffixes, cur_orig_toks, cur_v_idx) in enumerate(tqdm(grouped_input, desc="converting grouped inputs")):
		cur_toks = cur_toks.strip().split()
		cur_tok_l = [int(p) for p in cur_tok_l.strip().split()]
		cur_v_idx = [int(p) for p in cur_v_idx]

		tok_idx[ex_id, :len(cur_toks)] = np.array(tokenizer.convert_tokens_to_ids(cur_toks), dtype=int)
		v_length[ex_id] = len(cur_vn_labels)

		# note that the resulted actual seq length after subtoken collapsing could be different even within the same batch
		#	actual seq length is the origial sequence length
		#	seq length is the length after subword tokenization
		cur_sub2tok = sub2tok(opt, cur_tok_l)
		cur_sub2tok = pad(cur_sub2tok, opt.max_seq_l, [-1 for _ in range(opt.max_num_subtok)])
		sub2tok_idx[ex_id] = np.array(cur_sub2tok, dtype=int)

		orig_seq_length[ex_id] = len(cur_tok_l)
		seq_length[ex_id] = len(cur_toks)
		batch_keys[ex_id] = seq_length[ex_id]

		roleset_toks = [cur_roleset_ids[k] for k in range(len(cur_v_idx))]
		roleset_toks = [tokenizer.tokenize(sense) for sense in roleset_toks]
		roleset_toks = [p for subtoks in roleset_toks for p in subtoks]
		act_max_wp_l = max(act_max_wp_l, len(roleset_toks))
		roleset_toks = roleset_toks[:opt.max_wp_l-1] + [tokenizer.sep_token]
		wp_idx[ex_id, :len(roleset_toks)] = np.array(tokenizer.convert_tokens_to_ids(roleset_toks), dtype=int)

		cur_orig_toks = cur_orig_toks.strip().split()
		wpa = ['*' for _ in cur_orig_toks]
		for k, p in enumerate(cur_v_idx):
			wpa[p] = cur_roleset_ids[k]
		wpa = tokenizer.tokenize(' '.join(wpa))
		act_max_wpa_l = max(act_max_wpa_l, len(wpa))
		wpa = wpa[:opt.max_wpa_l-1] + [tokenizer.sep_token]
		wpa_idx[ex_id, :len(wpa)] = np.array(tokenizer.convert_tokens_to_ids(wpa), dtype=int)

		has_semlink = False
		for l_id, (vn, vnc, srl, sense) in enumerate(zip(cur_vn_labels, cur_vn_classes, cur_srl_labels, cur_roleset_suffixes)):
			num_prop += 1

			vn = vn.strip().split()
			srl = srl.strip().split()
			assert(len(vn) == len(srl))

			v_idx[ex_id, l_id] = cur_v_idx[l_id]
			roleset_id[ex_id, l_id] = roleset_indexer.convert(cur_roleset_ids[l_id])

			sense_id = roleset_suffix_indexer.convert(sense)
			roleset_suffix[ex_id, l_id] = sense_id

			vn_id = vn_label_indexer.convert_sequence(vn)
			vn_label[ex_id, l_id, :len(vn_id)] = np.array(vn_id, dtype=int)
			assert(vn_label[ex_id, l_id].sum() != 0)

			vnc_id = vn_class_indexer.convert(vnc)
			vn_class[ex_id, l_id] = vnc_id

			srl_id = srl_label_indexer.convert_sequence(srl)
			srl_label[ex_id, l_id, :len(srl_id)] = np.array(srl_id, dtype=int)			
			assert(srl_label[ex_id, l_id].sum() != 0)

			lemma = cur_roleset_ids[l_id].split('.')[0]
			frame_idx[ex_id, l_id] = frame_indexer.d[lemma]

			semlink_alignments = semlink_map[vnc_id, sense_id]
			semlink_l[ex_id, l_id] = (semlink_alignments[0] != -1).sum()
			semlink[ex_id, l_id] = semlink_alignments

			if semlink_l[ex_id, l_id] != 0:
				semlink_cnt += 1

		ex_id += 1

	if opt.shuffle == 1:
		rand_idx = np.random.permutation(ex_id)
		tok_idx = tok_idx[rand_idx]
		wp_idx = wp_idx[rand_idx]
		wpa_idx = wpa_idx[rand_idx]
		v_idx = v_idx[rand_idx]
		vn_label = vn_label[rand_idx]
		vn_class = vn_class[rand_idx]
		srl_label = srl_label[rand_idx]
		v_length = v_length[rand_idx]
		roleset_id = roleset_id[rand_idx]
		roleset_suffix = roleset_suffix[rand_idx]
		sub2tok_idx = sub2tok_idx[rand_idx]
		orig_seq_length = orig_seq_length[rand_idx]
		seq_length = seq_length[rand_idx]
		batch_keys = batch_keys[rand_idx]
		semlink = semlink[rand_idx]
		semlink_l = semlink_l[rand_idx]
		frame_idx = frame_idx[rand_idx]
		ex_idx = rand_idx

	# break up batches based on source/target lengths
	sorted_keys = sorted([(i, p) for i, p in enumerate(batch_keys)], key=lambda x: x[1])
	sorted_idx = [i for i, _ in sorted_keys]
	# rearrange examples	
	tok_idx = tok_idx[sorted_idx]
	wp_idx = wp_idx[sorted_idx]
	wpa_idx = wpa_idx[sorted_idx]
	v_idx = v_idx[sorted_idx]
	vn_label = vn_label[sorted_idx]
	vn_class = vn_class[sorted_idx]
	srl_label = srl_label[sorted_idx]
	v_l = v_length[sorted_idx]
	roleset_id = roleset_id[sorted_idx]
	roleset_suffix = roleset_suffix[sorted_idx]
	sub2tok_idx = sub2tok_idx[sorted_idx]
	seq_l = seq_length[sorted_idx]
	orig_seq_l = orig_seq_length[sorted_idx]
	semlink = semlink[sorted_idx]
	semlink_l = semlink_l[sorted_idx]
	frame_idx = frame_idx[sorted_idx]
	ex_idx = rand_idx[sorted_idx]

	mark_l = 0
	batch_location = [] #idx where sent length changes
	for j,i in enumerate(sorted_idx):
		if batch_keys[i] != mark_l:
			mark_l = seq_length[i]
			batch_location.append(j)
	if batch_location[-1] != len(tok_idx): 
		batch_location.append(len(tok_idx)-1)
	
	#get batch sizes
	curr_idx = 0
	batch_idx = [0]
	for i in range(len(batch_location)-1):
		end_location = batch_location[i+1]
		while curr_idx < end_location:
			curr_idx = min(curr_idx + opt.batch_size, end_location)
			batch_idx.append(curr_idx)

	batch_l = []
	seq_l_new = []
	for i in range(len(batch_idx)):
		end = batch_idx[i+1] if i < len(batch_idx)-1 else len(tok_idx)
		batch_l.append(end - batch_idx[i])
		seq_l_new.append(seq_l[batch_idx[i]])
		
		# sanity check
		for k in range(batch_idx[i], end):
			assert(seq_l[k] == seq_l_new[-1])
			assert(tok_idx[k, seq_l[k]:].sum() == 0)

	#
	print('act_max_wp_l', act_max_wp_l, 'cropped to', opt.max_wp_l)
	print('act_max_wpa_l', act_max_wpa_l, 'cropped to', opt.max_wpa_l)
	analysis(tok_idx, tokenizer.unk_token_id)

	# Write output
	f = h5py.File(output + '.hdf5', "w")		
	f["tok_idx"] = tok_idx
	f['wp_idx'] = wp_idx
	f['wpa_idx'] = wpa_idx
	f["v_idx"] = v_idx
	f["vn_label"] = vn_label
	f["vn_class"] = vn_class
	f["srl_label"] = srl_label
	f["v_l"] = v_l
	f["roleset_id"] = roleset_id
	f["roleset_suffix"] = roleset_suffix
	f['sub2tok_idx'] = sub2tok_idx
	f["seq_l"] = np.array(seq_l_new, dtype=int)
	f["orig_seq_l"] = orig_seq_l
	f["batch_l"] = batch_l
	f["batch_idx"] = batch_idx
	f["semlink"] = semlink
	f["semlink_l"] = semlink_l
	f["frame_idx"] = frame_idx
	f["frame_pool"] = frame_pool
	f['ex_idx'] = ex_idx
	print("saved {} batches ".format(len(f["batch_l"])))
	print("semlink was used {0} in {1} props".format(semlink_cnt, num_prop))
	f.close()


def sub2tok(opt, tok_l):
	acc = 0
	cur_sub2tok = []
	for l in tok_l:
		cur_sub2tok.append(pad([p for p in range(acc, acc+l)], opt.max_num_subtok, -1))
		assert(len(cur_sub2tok[-1]) <= opt.max_num_subtok)
		acc += l
	return cur_sub2tok


def unify_vn_label(label):
	if '-CO-' in label:
		label = label.replace('-CO-', '-CO_')
	if label.endswith('CAUSE'):
		label = label.replace('CAUSE', 'CAUSER')
	return label


def vnclass_inf(model, tok_idx, sub2tok_idx, toks, orig_toks, v_idx):
	vnclass = model.run_preprocessed(tok_idx, sub2tok_idx, toks, orig_toks, v_idx, v_type='vn')[0]
	return vnclass[0]

def sense_inf(model, tok_idx, sub2tok_idx, toks, orig_toks, v_idx):
	sense = model.run_preprocessed(tok_idx, sub2tok_idx, toks, orig_toks, v_idx, v_type='srl')[0]
	return sense[0]


def tokenize_and_write(opt, vnclass_model, sense_model, tokenizer, vn_srl_path, output, unify_vn=True):
	CLS, SEP = tokenizer.cls_token, tokenizer.sep_token
	all_orig_tok = []
	all_tok = []
	all_tok_l = []
	all_vn_label = []	# label sequence for vn example, where vnclass is replaced by B-V.
	all_vn_class = []	# vnclass from model prediction
	all_srl_label = []
	all_v_ids = []
	all_roleset_ids = []
	all_roleset_suffixes = []
	act_max_seq_l = 0

	mismatched_len_cnt = 0
	wrong_v_cnt = 0
	skip_cnt = 0
	with open(vn_srl_path, 'r') as f:
		for l in tqdm(f, desc="tokenizing {0}".format(vn_srl_path)):
			if l.strip() == '':
				continue
			
			parts = l.split('|||')
			assert(len(parts) == 4)

			sent, vn_labels, srl_labels, sense = parts[0].strip().split(), parts[1].strip().split(), parts[2].strip().split(), parts[3].strip()
			v_idx = int(sent[0])
			sent = sent[1:]	# removing the first trigger idx
			if unify_vn:
				vn_labels = [unify_vn_label(p) for p in vn_labels]
			vnclass = vn_labels[v_idx]
			assert(len(sent) == len(vn_labels))

			if len(vn_labels) != len(srl_labels):
				print(sent)
				print(vn_labels)
				print(srl_labels)
				mismatched_len_cnt += 1
				skip_cnt += 1
				continue

			sent_subtoks = [tokenizer.tokenize(w) for w in sent]
			tok_l = [len(subtoks) for subtoks in sent_subtoks]
			orig_toks = [w for w in sent]

			# find the position in the origial sequence for truncation
			trunc_orig_l = next((k for k, l in enumerate(np.cumsum(tok_l)) if l > opt.max_seq_l-2), len(tok_l))

			# take subtoks from truncated sequence
			sent_subtoks = sent_subtoks[:trunc_orig_l]
			orig_toks = orig_toks[:trunc_orig_l]
			tok_l = tok_l[:trunc_orig_l]
			vn_labels = vn_labels[:trunc_orig_l]
			srl_labels = srl_labels[:trunc_orig_l]
			toks = [p for subtoks in sent_subtoks for p in subtoks]	# flatterning

			# if after truncating, the labels are all O or the v_idx is out of range
			#	then skip this example
			if v_idx >= opt.max_seq_l-2 or v_idx >= len(vn_labels):
				skip_cnt += 1
				continue
			if sum([1 for k in vn_labels if k == 'O']) == len(toks) and sum([1 for k in srl_labels if k == 'O']) == len(toks):
				skip_cnt += 1
				continue

			act_max_seq_l = max(act_max_seq_l, len(toks))

			try:
				if vnclass_model:
					vnclass = vnclass_model(toks, orig_toks, v_idx, gold=vnclass)
				if sense_model:
					sense = sense_model(toks, orig_toks, v_idx, gold=sense)

				# if vnclass_model or sense_model:
					# sub2tok_idx = sub2tok(opt, tok_l)
					# sub2tok_idx = pad(sub2tok_idx, len(toks), [-1 for _ in range(opt.	max_num_subtok)])
					# tok_idx = tokenizer.convert_tokens_to_ids(toks)
	# 
					# if vnclass_model:
						# vnclass = vnclass_inf(vnclass_model, tok_idx, sub2tok_idx, toks, orig_toks, [v_idx])
	# 
					# if sense_model:
						# sense = sense_inf(sense_model, tok_idx, sub2tok_idx, toks, orig_toks, [v_idx])
			except:
				# print(tok_idx)
				# print(sub2tok_idx)
				# print(toks)
				# print(orig_toks)
				# print(v_idx)
				# printing stack trace
				traceback.print_exception(*sys.exc_info())
				exception_cnt += 1
				if exception_cnt >= 1000:
					assert(False)
				skip_cnt += 1
				continue

			if not is_any_vnclass(vnclass) or srl_labels[v_idx] != 'B-V':
				print(v_idx, sent)
				print(vn_labels)
				print(srl_labels)
				wrong_v_cnt += 1
				skip_cnt += 1
				continue

			# pad for CLS and SEP
			toks = [CLS] + toks + [SEP]
			vn_labels = ['O'] + vn_labels + ['O']
			srl_labels = ['O'] + srl_labels + ['O']
			tok_l = [1] + tok_l + [1]
			orig_toks = [CLS] + orig_toks + [SEP]
			v_idx += 1	# +1 to shift for prepended CLS token

			# force any V is set to O, and only keep the B-V at v_idx.
			vn_labels = ['O' if is_any_vnclass(l) else l for l in vn_labels]
			vn_labels[v_idx] = 'B-V'
			suffix = sense.split('.')[-1]

			all_tok.append(toks)
			all_tok_l.append(tok_l)
			all_vn_label.append(vn_labels)
			all_vn_class.append(vnclass)
			all_orig_tok.append(orig_toks)
			all_v_ids.append(v_idx)
			all_srl_label.append(srl_labels)
			all_roleset_ids.append(sense)
			all_roleset_suffixes.append(suffix)

	print('mismatched_len_cnt', mismatched_len_cnt)
	print('wrong_v_cnt', wrong_v_cnt)
	print('total skip cnt', skip_cnt)

	all_tok = [' '.join(p) for p in all_tok]
	all_tok_l = [' '.join([str(k) for k in p]) for p in all_tok_l]
	all_vn_label = [' '.join(p) for p in all_vn_label]
	all_vn_class = all_vn_class	# as-is
	all_orig_tok = [' '.join(p) for p in all_orig_tok]
	all_v_ids = [str(p) for p in all_v_ids]
	all_srl_label = [' '.join(p) for p in all_srl_label]
	all_roleset_ids = [str(p) for p in all_roleset_ids]
	all_roleset_suffixes = all_roleset_suffixes	# as-is


	print('act_max_seq_l: {0}'.format(act_max_seq_l))

	print('writing tokenized to {0}'.format(output + '.tok.txt'))
	with open(output + '.tok.txt', 'w') as f:
		for seq in all_tok:
			f.write(seq + '\n')

	print('writing token lengths to {0}'.format(output + '.tok_l.txt'))
	with open(output + '.tok_l.txt', 'w') as f:
		for seq in all_tok_l:
			f.write(seq + '\n')

	print('writing vn labels to {0}'.format(output + '.vn_label.txt'))
	with open(output + '.vn_label.txt', 'w') as f:
		for seq in all_vn_label:
			f.write(seq + '\n')

	print('writing vn classes to {0}'.format(output + '.vn_class.txt'))
	with open(output + '.vn_class.txt', 'w') as f:
		for seq in all_vn_class:
			f.write(seq + '\n')

	print('writing srl labels to {0}'.format(output + '.srl_label.txt'))
	with open(output + '.srl_label.txt', 'w') as f:
		for seq in all_srl_label:
			f.write(seq + '\n')

	print('writing roleset ids to {0}'.format(output + '.roleset_id.txt'))
	with open(output + '.roleset_id.txt', 'w') as f:
		for seq in all_roleset_ids:
			f.write(seq + '\n')

	print('writing roleset suffix to {0}'.format(output + '.roleset_suffix.txt'))
	with open(output + '.roleset_suffix.txt', 'w') as f:
		for seq in all_roleset_suffixes:
			f.write(seq + '\n')

	print('writing verb indices to {0}'.format(output + '.v_idx.txt'))
	with open(output + '.v_idx.txt', 'w') as f:
		for seq in all_v_ids:
			f.write(seq + '\n')
			
	print('writing original tokens to {0}'.format(output + '.orig_tok.txt'))
	with open(output + '.orig_tok.txt', 'w') as f:
		for seq in all_orig_tok:
			f.write(seq + '\n')

	return all_tok, all_tok_l, all_vn_label, all_vn_class, all_srl_label, all_roleset_ids, all_roleset_suffixes, all_orig_tok, all_v_ids


def load_tokenized(path):
	all_tok = []
	print('loading tokenized from {0}'.format(path + '.tok.txt'))
	with open(path + '.tok.txt', 'r') as f:
		for line in f:
			if line.strip() == '':
				continue
			all_tok.append(line.strip())

	all_tok_l = []
	print('loading token lengths from {0}'.format(path + '.tok_l.txt'))
	with open(path + '.tok_l.txt', 'r') as f:
		for line in f:
			if line.strip() == '':
				continue
			all_tok_l.append(line.strip())

	all_vn_label = []
	print('loading vn label from {0}'.format(path + '.vn_label.txt'))
	with open(path + '.vn_label.txt', 'r') as f:
		for line in f:
			if line.strip() == '':
				continue
			all_vn_label.append(line.strip())

	all_vn_class = []
	print('loading vn classes from {0}'.format(path + '.vn_class.txt'))
	with open(path + '.vn_class.txt', 'r') as f:
		for line in f:
			if line.strip() == '':
				continue
			all_vn_class.append(line.strip())

	all_srl_label = []
	print('loading srl labels from {0}'.format(path + '.srl_label.txt'))
	with open(path + '.srl_label.txt', 'r') as f:
		for line in f:
			if line.strip() == '':
				continue
			all_srl_label.append(line.strip())

	all_roleset_ids = []
	print('loading roleset ids from {0}'.format(path + '.roleset_id.txt'))
	with open(path + '.roleset_id.txt', 'r') as f:
		for line in f:
			if line.strip() == '':
				continue
			all_roleset_ids.append(line.strip())

	all_roleset_suffixes = []
	print('loading roleset suffix from {0}'.format(path + '.roleset_suffix.txt'))
	with open(path + '.roleset_suffix.txt', 'r') as f:
		for line in f:
			if line.strip() == '':
				continue
			all_roleset_suffixes.append(line.strip())

	all_v_ids = []
	print('loading verb indices from {0}'.format(path + '.v_idx.txt'))
	with open(path + '.v_idx.txt', 'r') as f:
		for line in f:
			if line.strip() == '':
				continue
			all_v_ids.append(line.strip())
		
	all_orig_tok = []	
	print('loading original tokens from {0}'.format(path + '.orig_tok.txt'))
	with open(path + '.orig_tok.txt', 'r') as f:
		for line in f:
			if line.strip() == '':
				continue
			all_orig_tok.append(line.strip())

	return all_tok, all_tok_l, all_vn_label, all_vn_class, all_srl_label, all_roleset_ids, all_roleset_suffixes, all_orig_tok, all_v_ids


def group_tokenized_with_labels(opt, toks, tok_l, vn_labels, vn_classes, srl_labels, roleset_ids, roleset_suffixes, orig_toks, v_ids, output):
	act_max_num_v = 0
	sent_map = {}
	for cur_toks, cur_tok_l, cur_vn_labels, cur_vn_classes, cur_srl_labels, cur_roleset_id, cur_roleset_suffix, cur_orig_toks, cur_v_idx in zip(
		toks, tok_l, vn_labels, vn_classes, srl_labels, roleset_ids, roleset_suffixes, orig_toks, v_ids):
		if cur_toks not in sent_map:
			sent_map[cur_toks] = []
		sent_map[cur_toks].append((cur_tok_l, cur_vn_labels, cur_vn_classes, cur_srl_labels, cur_roleset_id, cur_roleset_suffix, cur_orig_toks, cur_v_idx))

	rs = []
	for cur_toks, info in sent_map.items():
		cur_tok_l = info[0][0]
		act_max_num_v = max(act_max_num_v, len(info))
		# crop to max_num_v predicates
		info = info[:opt.max_num_v]

		cur_vn_labels = [p[1] for p in info]
		cur_vn_classes = [p[2] for p in info]
		cur_srl_labels = [p[3] for p in info]
		cur_roleset_ids = [p[4] for p in info]
		cur_roleset_suffixes = [p[5] for p in info]
		cur_orig_toks = info[0][6]
		cur_v_idx = [p[7] for p in info]
		# (tokens, token length, vn labels, srl labels, orig tokens, v idx)
		rs.append((cur_toks, cur_tok_l, cur_vn_labels, cur_vn_classes, cur_srl_labels, cur_roleset_ids, cur_roleset_suffixes, cur_orig_toks, cur_v_idx))

	print('writing grouped original tokens to {0}'.format(output + '.orig_tok_grouped.txt'))
	with open(output + '.orig_tok_grouped.txt', 'w') as f:
		for row in rs:
			f.write(row[7] + '\n')

	print('maximum number of predicates: {0}, cropped to {1}'.format(act_max_num_v, opt.max_num_v))
	return rs 	# (toks, tok_l, list of labels)


def load(path):
	all_lines = []
	with open(path, 'r') as f:
		for l in f:
			if l.rstrip() == '':
				continue
			all_lines.append(l.strip())
	return all_lines


def analysis(toks, unk_idx):
	unk_cnt = 0
	for row in toks:
		if len([1 for idx in row if idx == unk_idx]) != 0:
			unk_cnt += 1
	print('{0} examples has token unk.'.format(unk_cnt))


def get_semlink_map(opt, vn_label_indexer, vn_class_indexer, srl_label_indexer, roleset_suffix_indexer, semlink):
	semlink_map = np.ones((
		len(vn_class_indexer.d),
		len(roleset_suffix_indexer.d),
		2,
		opt.max_num_semlink), dtype=np.int32) * -1
	valid_cnt = 0
	act_max_num_semlink = 0 	# the upper bound we found in semlink
	for (vnc, suffix), pairs in tqdm(semlink.items(), desc="converting semlink pool"):
		vnc = 'B-' + vnc
		if vnc not in vn_class_indexer.d or suffix not in roleset_suffix_indexer.d:
			continue
		vnc_idx = vn_class_indexer.convert(vnc)
		suffix_idx = roleset_suffix_indexer.convert(suffix)

		idx_pairs = []
		for r, a, in pairs:
			br, ba = 'B-' + r, 'B-' + a
			# ir, ia = 'I-' + r, 'I-' + a
			bcr, bca = 'B-C-' + r, 'B-C-' + a
			# icr, ica = 'I-C-' + r, 'I-C-' + a
			brr, bra = 'B-R-' + r, 'B-R-' + a
			# irr, ira = 'I-R-' + r, 'I-R-' + a
			if br in srl_label_indexer.d and ba in vn_label_indexer.d:
				idx_pairs.append((srl_label_indexer.d[br], vn_label_indexer.d[ba]))

			if bcr in srl_label_indexer.d and bca in vn_label_indexer.d:
				idx_pairs.append((srl_label_indexer.d[bcr], vn_label_indexer.d[bca]))

			if brr in srl_label_indexer.d and bra in vn_label_indexer.d:
				idx_pairs.append((srl_label_indexer.d[brr], vn_label_indexer.d[bra]))

			# if ir in srl_label_indexer.d and ia in vn_label_indexer.d:
			# 	idx_pairs.append((srl_label_indexer.d[ir], vn_label_indexer.d[ia]))
# 
			# if icr in srl_label_indexer.d and ica in vn_label_indexer.d:
			# 	idx_pairs.append((srl_label_indexer.d[icr], vn_label_indexer.d[ica]))
# 
			# if irr in srl_label_indexer.d and ira in vn_label_indexer.d:
			# 	idx_pairs.append((srl_label_indexer.d[irr], vn_label_indexer.d[ira]))

		act_max_num_semlink = max(act_max_num_semlink, len(idx_pairs))
		idx_pairs = np.asarray(idx_pairs, dtype=np.int32).transpose((1, 0))
		idx_pairs = idx_pairs[:, :opt.max_num_semlink]
		semlink_map[vnc_idx, suffix_idx, :, :idx_pairs.shape[1]] = idx_pairs

		if (idx_pairs != -1).sum() != 0:
			valid_cnt += 1

	print(f'act_max_num_semlink {act_max_num_semlink}, cropped to {opt.max_num_semlink}')
	print(f'found {valid_cnt} valid semlink entries.')
	return semlink_map


def analyze_arg_role_map(arg_role_map, arg_role_map_inv, vn_labels, srl_labels):
	vio_map = defaultdict(lambda: 0)
	vio_cnt = 0
	sat_cnt = 0
	for vn_seq, srl_seq in zip(vn_labels, srl_labels):
		vn_seq = vn_seq.strip().split()
		srl_seq = srl_seq.strip().split()
		for v, s in zip(vn_seq, srl_seq):
			v_prefix, v_root = parse_label(v)
			s_prefix, s_root = parse_label(s)
			if v_prefix or s_prefix:
				if (v_root, s_root) not in arg_role_map:
					vio_map[(v_root, s_root)] += 1
					vio_cnt += 1
				else:
					sat_cnt += 1
	print("Number of arg-role pairs violating predefined cross label pairs:", vio_cnt)
	if vio_map:
		print(vio_map)
	print("Number of arg-role pairs satisfying predefined cross label pairs:", sat_cnt)


def analyze_semlink(semlink_map, vn_class_indexer, roleset_suffix_indexer,
					vn_label_indexer, srl_label_indexer, vnclass, roleset_suffixes, vn_labels, srl_labels):
	vn_inv_map = ['O' for _ in range(len(vn_label_indexer.d))]
	for k, v in vn_label_indexer.d.items():
		vn_inv_map[v] = k
	srl_inv_map = ['O' for _ in range(len(srl_label_indexer.d))]
	for k, v in srl_label_indexer.d.items():
		srl_inv_map[v] = k
	
	vio_cnt = 0
	sat_cnt = 0
	pred_vio_cnt = 0
	vio_map = defaultdict(lambda: defaultdict(lambda: 0))

	def record_vio(vio_map, vnc, suffix, role, arg):
		if arg.startswith('B-'):
			vio_map[(vnc, suffix)][(role, arg)] += 1

	for i, (vnc, suffix, vn_seq, srl_seq) in enumerate(zip(vnclass, roleset_suffixes, vn_labels, srl_labels)):
		vnc_idx = vn_class_indexer.convert(vnc)
		sense_idx = roleset_suffix_indexer.convert(suffix)
		vn_seq = vn_seq.strip().split()
		srl_seq = srl_seq.strip().split()
		pairs = semlink_map[vnc_idx, sense_idx, :, :]
		pairs = [(pairs[0, k], pairs[1,k]) for k in range(pairs.shape[1]) if pairs[0, k] != -1 and pairs[1, k] != -1]
		pretty_pairs = [(srl_inv_map[r], vn_inv_map[a]) for r, a in pairs]

		pred_has_vio = False
		for arg, role in zip(vn_seq, srl_seq):
			if arg not in vn_label_indexer.d or role not in srl_label_indexer.d:
				continue
			arg_idx = vn_label_indexer.d[arg]
			role_idx = srl_label_indexer.d[role]
			if arg_idx == 0 or role_idx == 0:
				continue
			# if pairs is not empty
			if any(True for p in pairs if p[0] == role_idx) or any(True for p in pairs if p[1] == arg_idx):
				if (role_idx, arg_idx) not in pairs:
					record_vio(vio_map, vnc, suffix, role, arg)
					if arg.startswith('B-'):
						vio_cnt += 1
						pred_has_vio = True
						continue
			if arg.startswith('B-'):
				sat_cnt += 1
		if pred_has_vio:
			pred_vio_cnt += 1

	print("Number of label pairs satisfies semlink:", sat_cnt)
	print("Number of label pairs violates semlink:", vio_cnt)
	print(f"predicate semlink violation: {pred_vio_cnt}/{len(vn_labels)}")
	if vio_map:
		print("Clustered label pairs that violates semlink:")
		for key, pair_cnt in vio_map.items():
			# print(f'vnclass: {key[0]}, roleset suffix: {key[1]}')
			for pair, cnt in pair_cnt.items():
				if cnt > 5:
					# dump correction code to add to manual_fix_for_semlink
					print(f"{cnt:4d} semlink[('{key[0][2:]}', '{key[1]}')].append(('{pair[0][2:]}', '{pair[1][2:]}'))")


def process(opt):
	tokenizer = get_tokenizer(opt.transformer_type)

	if opt.indexer != opt.dir:
		vn_label_indexer = Indexer.load_from(opt.indexer + '.vn_label.dict', oov_token='O')
		vn_class_indexer = Indexer.load_from(opt.indexer + '.vn_class.dict', oov_token='O')
		srl_label_indexer = Indexer.load_from(opt.indexer + '.srl_label.dict', oov_token='O')
		roleset_indexer = Indexer.load_from(opt.indexer + '.roleset_label.dict', oov_token='O')
		roleset_suffix_indexer = Indexer.load_from(opt.indexer + '.roleset_suffix.dict', oov_token='O')
		frame_indexer = Indexer.load_from(opt.indexer + 'frame.dict', oov_token='#')
	else:
		vn_label_indexer = Indexer(symbols=["O"], oov_token='O')
		vn_class_indexer = Indexer(symbols=["O"], oov_token='O')
		srl_label_indexer = Indexer(symbols=["O"], oov_token='O')
		roleset_indexer = Indexer(symbols=["O"], oov_token='O')
		roleset_suffix_indexer = Indexer(symbols=["O"], oov_token='O')
		frame_indexer = Indexer(symbols=["#"], oov_token='#')

	arg_role_map, arg_role_map_inv = load_arg_role(opt.arg_role_map)
	frameset = load_frameset(opt.frameset)

	#### tokenize
	train_vn_srl = path_prefix(opt.data, 'train', opt.part) + '.vn_srl.txt'
	val_vn_srl = path_prefix(opt.data, 'val', opt.part) + '.vn_srl.txt'
	test1_vn_srl = path_prefix(opt.data, 'test1', opt.part) + '.vn_srl.txt'

	vnclass_model = None
	sense_model = None


	if opt.load_tokenized == 0:
		train_output = path_prefix(opt.output, 'train', opt.part)
		train_toks, train_tok_l, train_vn_labels, train_vn_classes, train_srl_labels, train_roleset_ids, train_roleset_suffixes, train_orig_toks, train_v_ids = tokenize_and_write(opt, vnclass_model, sense_model, tokenizer, train_vn_srl, train_output)
	
		val_output = path_prefix(opt.output, 'val', opt.part)
		val_toks, val_tok_l, val_vn_labels, val_vn_classes, val_srl_labels, val_roleset_ids, val_roleset_suffixes, val_orig_toks, val_v_ids = tokenize_and_write(opt, vnclass_model, sense_model, tokenizer, val_vn_srl, val_output)
	
		test1_output = path_prefix(opt.output, 'test1', opt.part)
		test1_toks, test1_tok_l, test1_vn_labels, test1_vn_classes, test1_srl_labels, test1_roleset_ids, test1_roleset_suffixes, test1_orig_toks, test1_v_ids = tokenize_and_write(opt, vnclass_model, sense_model, tokenizer, test1_vn_srl, test1_output)
	else:
		train_output = path_prefix(opt.output, 'train', opt.part)
		train_toks, train_tok_l, train_vn_labels, train_vn_classes, train_srl_labels, train_roleset_ids, train_roleset_suffixes, train_orig_toks, train_v_ids = load_tokenized(train_output)
		val_output = path_prefix(opt.output, 'val', opt.part)
		val_toks, val_tok_l, val_vn_labels, val_vn_classes, val_srl_labels, val_roleset_ids, val_roleset_suffixes, val_orig_toks, val_v_ids = load_tokenized(val_output)
		test1_output = path_prefix(opt.output, 'test1', opt.part)
		test1_toks, test1_tok_l, test1_vn_labels, test1_vn_classes, test1_srl_labels, test1_roleset_ids, test1_roleset_suffixes, test1_orig_toks, test1_v_ids = load_tokenized(test1_output)


	print("Analyzing arg_role_map...")
	analyze_arg_role_map(arg_role_map, arg_role_map_inv, train_vn_labels, train_srl_labels)

	print("Number of predicates in training: {}".format(len(train_toks)))
	print("Number of predicates in val: {}".format(len(val_toks)))
	print("Number of predicates in test1: {}".format(len(test1_toks)))

	if opt.aug_indexer == 1:
		print("First pass through data to get label vocab...")
		_ = make_label_vocab1(opt, frame_indexer, [lemma for lemma, _ in frameset.items()], count=False)

		_ = make_label_vocab2(opt, vn_label_indexer, train_vn_labels, count=True)
		_ = make_label_vocab2(opt, srl_label_indexer, train_srl_labels, count=True)
		_ = make_label_vocab1(opt, vn_class_indexer, train_vn_classes, count=True)
		_ = make_label_vocab1(opt, roleset_indexer, train_roleset_ids, count=True)
		_ = make_label_vocab1(opt, roleset_suffix_indexer, train_roleset_suffixes, count=True)
		_ = make_label_vocab1(opt, frame_indexer, [p.split('.')[0] for p in train_roleset_ids], count=True)

		_ = make_label_vocab2(opt, vn_label_indexer, val_vn_labels, count=True)
		_ = make_label_vocab2(opt, srl_label_indexer, val_srl_labels, count=True)
		_ = make_label_vocab1(opt, vn_class_indexer, val_vn_classes, count=True)
		_ = make_label_vocab1(opt, roleset_indexer, val_roleset_ids, count=True)
		_ = make_label_vocab1(opt, roleset_suffix_indexer, val_roleset_suffixes, count=True)
		_ = make_label_vocab1(opt, frame_indexer, [p.split('.')[0] for p in val_roleset_ids], count=True)
	
		_ = make_label_vocab2(opt, vn_label_indexer, test1_vn_labels, count=False)	# no counting on test set
		_ = make_label_vocab2(opt, srl_label_indexer, test1_srl_labels, count=False)
		_ = make_label_vocab1(opt, vn_class_indexer, test1_vn_classes, count=False)
		_ = make_label_vocab1(opt, roleset_indexer, test1_roleset_ids, count=False)
		_ = make_label_vocab1(opt, roleset_suffix_indexer, test1_roleset_suffixes, count=False)
		_ = make_label_vocab1(opt, frame_indexer, [p.split('.')[0] for p in test1_roleset_ids], count=False)

		# also record any extra suffix defined in frameset
		for _, p in frameset.items():
			suffix, _ = p
			roleset_suffix_indexer.register_all_words([suffix], count=False)

		indexer_output = f'{opt.data}.{opt.part}'
		vn_label_indexer.write(indexer_output + ".vn_label.dict")
		vn_class_indexer.write(indexer_output + ".vn_class.dict")
		srl_label_indexer.write(indexer_output + ".srl_label.dict")
		roleset_indexer.write(indexer_output + ".roleset_label.dict")
		roleset_suffix_indexer.write(indexer_output + ".roleset_suffix.dict")
		frame_indexer.write(indexer_output + ".frame.dict")
		print('vn label size: {}'.format(len(vn_label_indexer.d)))
		print('vn class size: {}'.format(len(vn_class_indexer.d)))
		print('srl label size: {}'.format(len(srl_label_indexer.d)))
		print('roleset label size: {}'.format(len(roleset_indexer.d)))
		print('roleset suffix size: {}'.format(len(roleset_suffix_indexer.d)))
		print('frame size: {}'.format(len(frame_indexer.d)))

	# frameset
	frame_pool = extract_frame_pool(roleset_suffix_indexer, srl_label_indexer, frame_indexer, frameset)

	sense_vn_map, suffix_vn_map = load_semlink(opt.data)
	semlink = extract_semlink(sense_vn_map, suffix_vn_map)
	semlink = manual_fix_for_semlink(semlink, opt.collapse_vc)
	semlink_map = get_semlink_map(opt, vn_label_indexer, vn_class_indexer, srl_label_indexer, roleset_suffix_indexer, semlink)

	print('analyzing semlink on train set...')
	analyze_semlink(semlink_map, vn_class_indexer, roleset_suffix_indexer, vn_label_indexer, srl_label_indexer,
		train_vn_classes, train_roleset_suffixes, train_vn_labels, train_srl_labels)

	print('analyzing semlink on val set...')
	analyze_semlink(semlink_map, vn_class_indexer, roleset_suffix_indexer, vn_label_indexer, srl_label_indexer,
		val_vn_classes, val_roleset_suffixes, val_vn_labels, val_srl_labels)

	convert(opt, tokenizer, vn_label_indexer, vn_class_indexer, srl_label_indexer, roleset_indexer, roleset_suffix_indexer, frame_indexer, frame_pool,
		val_toks, val_tok_l, val_vn_labels, val_vn_classes, val_srl_labels, val_roleset_ids, val_roleset_suffixes, val_orig_toks, val_v_ids,
		semlink_map, val_output)

	convert(opt, tokenizer, vn_label_indexer, vn_class_indexer, srl_label_indexer, roleset_indexer, roleset_suffix_indexer, frame_indexer, frame_pool,
		train_toks, train_tok_l, train_vn_labels, train_vn_classes, train_srl_labels, train_roleset_ids, train_roleset_suffixes, train_orig_toks, train_v_ids,
		semlink_map, train_output)

	convert(opt, tokenizer, vn_label_indexer, vn_class_indexer, srl_label_indexer, roleset_indexer, roleset_suffix_indexer, frame_indexer, frame_pool,
		test1_toks, test1_tok_l, test1_vn_labels, test1_vn_classes, test1_srl_labels, test1_roleset_ids, test1_roleset_suffixes, test1_orig_toks, test1_v_ids,
		semlink_map, test1_output)


def path_prefix(data, split, part):
	return f'{data}.{split}.{part}'

	
def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--data', help="Prefix to data name", default='upc_vn2')
	parser.add_argument('--part', help="Part of the split",default = "matched")
	parser.add_argument('--dir', help="Path to the data dir",default = "./data/vn_modular/")
	parser.add_argument('--transformer_type', help="The type of transformer encoder from huggingface, eg. roberta-base",default = "roberta-base")

	parser.add_argument('--arg_role_map', help="The path to valid arg-role mapping for semlink", default = "upc_vn_arg_role.txt")
	parser.add_argument('--indexer', help="Path to existing indexer dict (optional)", default="")
	parser.add_argument('--aug_indexer', help="Whether to augment indexer", type=int, default=1)
	parser.add_argument('--frameset', help="Path to extracted role set.", default = "frameset.txt")
	
	parser.add_argument('--batch_size', help="Size of each minibatch.", type=int, default=8)
	parser.add_argument('--max_seq_l', help="Maximal sequence length", type=int, default=300)
	parser.add_argument('--max_wp_l', help="Maximal length for predicate tokens to append to input", type=int, default=100)
	parser.add_argument('--max_wpa_l', help="Maximal length for aligned predicate tokens to append to input", type=int, default=200)
	parser.add_argument('--max_num_v', help="Maximal number of predicate in a sentence", type=int, default=16)
	parser.add_argument('--max_num_subtok', help="Maximal number subtokens in a word", type=int, default=8)
	parser.add_argument('--max_num_semlink', help="Maximal number of semlink role-argument mapping for a predicate.", type=int, default=24)
	parser.add_argument('--load_tokenized', help="Whether to load tokenized instead of doing tokenization", type=int, default=0)
	parser.add_argument('--output', help="Prefix of the output file names. ", type=str, default="upc_vn2")
	parser.add_argument('--shuffle', help="If = 1, shuffle sentences before sorting (based on source length).", type=int, default=1)
	parser.add_argument('--collapse_vc', help="Whether collapse vn class id.", type=int, default=0)
	parser.add_argument('--seed', help="The random seed", type=int, default=1)

	opt = parser.parse_args(arguments)

	opt.data = opt.dir + opt.data
	opt.output = opt.dir + opt.output
	opt.arg_role_map = opt.dir + opt.arg_role_map
	opt.frameset = opt.dir + opt.frameset

	opt.indexer = opt.dir + opt.indexer
	

	process(opt)

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
