import sys
import argparse
import numpy as np
import h5py
from collections import defaultdict
from transformers import *
import json
from tqdm import tqdm
from . import extract
from util.util import Indexer


def load_semlink(pb_vn_path):
	rs = {}
	with open(pb_vn_path, 'r') as f:
		pb_vn = json.load(f)
		undef_cnt = 0
		for sense, vn_ls in tqdm(pb_vn.items(), desc="loading pb-vn mapping from {0}".format(pb_vn_path)):
			if len(vn_ls) == 0:
				undef_cnt += 1
				continue

			for vn in vn_ls:
				vc_id = vn['vnclass']

				pairs = []
				for role, arg in vn['args']:
					if extract.fix_role(role) and arg != 'NONE':
						pairs.append((extract.fix_role(role), arg))
				if pairs:
					rs[(vc_id, sense)] = pairs
	print(f'loaded {len(rs)} pb-vn entries.')
	return rs


def get_tokenizer(key):
	model_map={"bert-base-uncased": (BertModel, BertTokenizer),
		"roberta-base": (RobertaModel, RobertaTokenizer)}
	model_cls, tokenizer_cls = model_map[key]
	print('loading tokenizer: {0}'.format(key))
	tokenizer = tokenizer_cls.from_pretrained(key)
	return tokenizer


def get_unk_idx(key):
	unk_map={"bert-base-uncased": 100,
		"roberta-base": 3}
	return unk_map[key]

			
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


def convert(opt, tokenizer, vn_label_indexer, srl_label_indexer, roleset_indexer, roleset_suffix_indexer,
		toks, tok_l, vn_labels, srl_labels, roleset_ids, roleset_suffixes, orig_toks, v_ids,
		semlink, semlink_map, semlink_pool,
		tokenizer_output, output):
	np.random.seed(opt.seed)

	grouped_input = group_tokenized_with_labels(opt, toks, tok_l, vn_labels, srl_labels, roleset_ids, roleset_suffixes, orig_toks, v_ids, tokenizer_output)
	num_ex = len(grouped_input)

	tok_idx = np.zeros((num_ex, opt.max_seq_l), dtype=np.int32)
	sub2tok_idx = np.zeros((num_ex, opt.max_seq_l, opt.max_num_subtok), dtype=np.int32) - 1	# empty is -1
	v_idx = np.zeros((num_ex, opt.max_num_v), dtype=np.int32)
	vn_label = np.zeros((num_ex, opt.max_num_v, opt.max_seq_l), dtype=np.int32)
	srl_label = np.zeros((num_ex, opt.max_num_v, opt.max_seq_l), dtype=np.int32)
	v_length = np.zeros((num_ex,), dtype=np.int32)	# number of v
	roleset_id = np.zeros((num_ex, opt.max_num_v), dtype=np.int32)
	roleset_suffix = np.zeros((num_ex, opt.max_num_v), dtype=np.int32)
	seq_length = np.zeros((num_ex,), dtype=np.int32)
	orig_seq_length = np.zeros((num_ex,), dtype=np.int32)
	ex_idx = np.zeros(num_ex, dtype=np.int32)
	batch_keys = np.array([None for _ in range(num_ex)])

	# each predicate has 2 lines mapping for semlink, first line is semantic role, second vn arg.
	sml = np.zeros((num_ex, opt.max_num_v, 2, opt.max_num_semlink), dtype=np.int32)
	sml_l = np.zeros((num_ex, opt.max_num_v), dtype=np.int32)

	ex_id = 0
	semlink_used = 0
	for _, (cur_toks, cur_tok_l, cur_vn_labels, cur_srl_labels, cur_roleset_ids, cur_roleset_suffixes, cur_orig_toks, cur_v_idx) in enumerate(tqdm(grouped_input, desc="converting grouped inputs")):
		cur_toks = cur_toks.strip().split()
		cur_tok_l = [int(p) for p in cur_tok_l.strip().split()]
		cur_v_idx = [int(p) for p in cur_v_idx]

		tok_idx[ex_id, :len(cur_toks)] = np.array(tokenizer.convert_tokens_to_ids(cur_toks), dtype=int)
		v_length[ex_id] = len(cur_vn_labels)

		# note that the resulted actual seq length after subtoken collapsing could be different even within the same batch
		#	actual seq length is the origial sequence length
		#	seq length is the length after subword tokenization
		acc = 0
		cur_sub2tok = []
		for l in cur_tok_l:
			cur_sub2tok.append(pad([p for p in range(acc, acc+l)], opt.max_num_subtok, -1))
			assert(len(cur_sub2tok[-1]) <= opt.max_num_subtok)
			acc += l
		cur_sub2tok = pad(cur_sub2tok, opt.max_seq_l, [-1 for _ in range(opt.max_num_subtok)])
		sub2tok_idx[ex_id] = np.array(cur_sub2tok, dtype=int)

		orig_seq_length[ex_id] = len(cur_tok_l)
		seq_length[ex_id] = len(cur_toks)
		batch_keys[ex_id] = seq_length[ex_id]

		has_semlink = False
		for l_id, (vn, srl) in enumerate(zip(cur_vn_labels, cur_srl_labels)):
			vn = vn.strip().split()
			srl = srl.strip().split()
			assert(len(vn) == len(srl))
			#v_idx[ex_id, l_id] = cur_labels.index('B-V')	# there MUST be a B-V since we are reading from role labels
			v_idx[ex_id, l_id] = cur_v_idx[l_id]
			roleset_id[ex_id, l_id] = roleset_indexer.convert(cur_roleset_ids[l_id])
			roleset_suffix[ex_id, l_id] = roleset_suffix_indexer.convert(cur_roleset_suffixes[l_id])

			vn_id = vn_label_indexer.convert_sequence(vn)
			vn_label[ex_id, l_id, :len(vn_id)] = np.array(vn_id, dtype=int)
			assert(vn_label[ex_id, l_id].sum() != 0)

			srl_id = srl_label_indexer.convert_sequence(srl)
			srl_label[ex_id, l_id, :len(srl_id)] = np.array(srl_id, dtype=int)			
			assert(srl_label[ex_id, l_id].sum() != 0)

			vc = vn[cur_v_idx[l_id]].split('B-')[1]
			sense = cur_roleset_ids[l_id]
			if (vc, sense) in semlink:
				pairs = semlink[(vc, sense)]
				pairs = pairs[:opt.max_num_semlink]
				role_idx = [srl_label_indexer.d['B-'+r] for r, _ in pairs]
				arg_idx = [vn_label_indexer.d['B-'+a] for _, a in pairs]
				sml[ex_id, l_id, 0, :len(pairs)] = np.array(role_idx, dtype=int)
				sml[ex_id, l_id, 1, :len(pairs)] = np.array(arg_idx, dtype=int)
				sml_l[ex_id, l_id] = len(pairs)
				has_semlink = True

		ex_id += 1
		if has_semlink:
			semlink_used += 1

	if opt.shuffle == 1:
		rand_idx = np.random.permutation(ex_id)
		tok_idx = tok_idx[rand_idx]
		v_idx = v_idx[rand_idx]
		vn_label = vn_label[rand_idx]
		srl_label = srl_label[rand_idx]
		v_length = v_length[rand_idx]
		roleset_id = roleset_id[rand_idx]
		roleset_suffix = roleset_suffix[rand_idx]
		sub2tok_idx = sub2tok_idx[rand_idx]
		orig_seq_length = orig_seq_length[rand_idx]
		seq_length = seq_length[rand_idx]
		batch_keys = batch_keys[rand_idx]
		sml = sml[rand_idx]
		sml_l = sml_l[rand_idx]
		ex_idx = rand_idx

	# break up batches based on source/target lengths
	sorted_keys = sorted([(i, p) for i, p in enumerate(batch_keys)], key=lambda x: x[1])
	sorted_idx = [i for i, _ in sorted_keys]
	# rearrange examples	
	tok_idx = tok_idx[sorted_idx]
	v_idx = v_idx[sorted_idx]
	vn_label = vn_label[sorted_idx]
	srl_label = srl_label[sorted_idx]
	v_l = v_length[sorted_idx]
	roleset_id = roleset_id[sorted_idx]
	roleset_suffix = roleset_suffix[sorted_idx]
	sub2tok_idx = sub2tok_idx[sorted_idx]
	seq_l = seq_length[sorted_idx]
	orig_seq_l = orig_seq_length[sorted_idx]
	sml = sml[sorted_idx]
	sml_l = sml_l[sorted_idx]
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
	analysis(tok_idx, tokenizer.unk_token_id)

	# Write output
	f = h5py.File(output, "w")		
	f["tok_idx"] = tok_idx
	f["v_idx"] = v_idx
	f["vn_label"] = vn_label
	f["srl_label"] = srl_label
	f["v_l"] = v_l
	f["roleset_id"] = roleset_id
	f["roleset_suffix"] = roleset_suffix
	f['sub2tok_idx'] = sub2tok_idx
	f["seq_l"] = np.array(seq_l_new, dtype=int)
	f["orig_seq_l"] = orig_seq_l
	f["batch_l"] = batch_l
	f["batch_idx"] = batch_idx
	f["semlink"] = sml
	f["semlink_l"] = sml_l
	f['semlink_map'] = semlink_map
	f['semlink_pool'] = semlink_pool
	f['ex_idx'] = ex_idx
	print("saved {} batches ".format(len(f["batch_l"])))
	print("semlink was used {} times".format(semlink_used))
	f.close()  

def tokenize_and_write(opt, tokenizer, vn_path, srl_path, output):
	CLS, SEP = tokenizer.cls_token, tokenizer.sep_token
	all_orig_tok = []
	all_tok = []
	all_tok_l = []
	all_vn_label = []
	all_srl_label = []
	all_v_ids = []
	all_roleset_ids = []
	all_roleset_suffixes = []
	act_max_seq_l = 0
	with open(vn_path, 'r') as f:
		for l in tqdm(f, desc="tokenizing {0}".format(vn_path)):
			if l.strip() == '':
				continue
			
			parts = l.split('|||')
			assert(len(parts) == 2 or len(parts) == 3)	# if parts has 3: words ||| labels ||| roleset id; if parts has 2: words ||| labels

			sent, labels = parts[0].strip().split(), parts[1].strip().split()
			v_idx = int(sent[0])
			sent = sent[1:]	# removing the first trigger idx
			assert(len(sent) == len(labels))

			sent_subtoks = [tokenizer.tokenize(w) for w in sent]
			tok_l = [len(subtoks) for subtoks in sent_subtoks]
			orig_toks = [w for w in sent]

			# find the position in the origial sequence for truncation
			trunc_orig_l = next((k for k, l in enumerate(np.cumsum(tok_l)) if l > opt.max_seq_l-2), len(tok_l))

			# take subtoks from truncated sequence
			sent_subtoks = sent_subtoks[:trunc_orig_l]
			orig_toks = orig_toks[:trunc_orig_l]
			tok_l = tok_l[:trunc_orig_l]
			labels = labels[:trunc_orig_l]
			toks = [p for subtoks in sent_subtoks for p in subtoks]	# flatterning

			# if after truncating, the labels are all O or the v_idx is out of range
			#	then skip this example
			if v_idx >= opt.max_seq_l-2 or sum([1 for k in labels if k == 'O']) == len(labels):
				continue

			act_max_seq_l = max(act_max_seq_l, len(toks))

			# pad for CLS and SEP
			toks = [CLS] + toks + [SEP]
			labels = ['O'] + labels + ['O']
			tok_l = [1] + tok_l + [1]
			orig_toks = [CLS] + orig_toks + [SEP]
			v_idx += 1	# +1 to shift for prepended CLS token

			all_tok.append(' '.join(toks))
			all_tok_l.append(' '.join([str(p) for p in tok_l]))
			all_vn_label.append(' '.join(labels))
			all_orig_tok.append(' '.join(orig_toks))
			all_v_ids.append(str(v_idx))

	with open(srl_path, 'r') as f:
		for l in tqdm(f, desc="tokenizing {0}".format(srl_path)):
			if l.strip() == '':
				continue
			
			parts = l.split('|||')
			# orig toks ||| labels ||| roleset id
			assert(len(parts) == 3)

			sent, labels, roleset_id = parts[0].strip().split(), parts[1].strip().split(), parts[2].strip()
			v_idx = int(sent[0])
			sent = sent[1:]	# removing the first trigger idx
			assert(len(sent) == len(labels))

			sent_subtoks = [tokenizer.tokenize(w) for w in sent]
			tok_l = [len(subtoks) for subtoks in sent_subtoks]
			orig_toks = [w for w in sent]

			# find the position in the origial sequence for truncation
			trunc_orig_l = next((k for k, l in enumerate(np.cumsum(tok_l)) if l > opt.max_seq_l-2), len(tok_l))

			# take subtoks from truncated sequence
			sent_subtoks = sent_subtoks[:trunc_orig_l]
			orig_toks = orig_toks[:trunc_orig_l]
			tok_l = tok_l[:trunc_orig_l]
			labels = labels[:trunc_orig_l]
			toks = [p for subtoks in sent_subtoks for p in subtoks]	# flatterning

			# if after truncating, the labels are all O or the v_idx is out of range
			#	then skip this example
			if v_idx >= opt.max_seq_l-2 or sum([1 for k in labels if k == 'O']) == len(labels):
				continue

			act_max_seq_l = max(act_max_seq_l, len(toks))

			# pad for CLS and SEP
			labels = ['O'] + labels + ['O']
			all_srl_label.append(' '.join(labels))
			all_roleset_ids.append(str(roleset_id))
			all_roleset_suffixes.append(str(roleset_id).split('.')[-1])

	assert(len(all_vn_label) == len(all_srl_label))


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


	return all_tok, all_tok_l, all_vn_label, all_srl_label, all_roleset_ids, all_roleset_suffixes, all_orig_tok, all_v_ids


def group_tokenized_with_labels(opt, toks, tok_l, vn_labels, srl_labels, roleset_ids, roleset_suffixes, orig_toks, v_ids, output):
	act_max_num_v = 0
	sent_map = {}
	for cur_toks, cur_tok_l, cur_vn_labels, cur_srl_labels, cur_roleset_id, cur_roleset_suffix, cur_orig_toks, cur_v_idx in zip(
		toks, tok_l, vn_labels, srl_labels, roleset_ids, roleset_suffixes, orig_toks, v_ids):
		if cur_toks not in sent_map:
			sent_map[cur_toks] = []
		if len(sent_map[cur_toks]) >= opt.max_num_v:
			continue
		sent_map[cur_toks].append((cur_tok_l, cur_vn_labels, cur_srl_labels, cur_roleset_id, cur_roleset_suffix, cur_orig_toks, cur_v_idx))

	rs = []
	for cur_toks, info in sent_map.items():
		cur_tok_l = info[0][0]
		act_max_num_v = max(act_max_num_v, len(info))
		cur_vn_labels = [p[1] for p in info]
		cur_srl_labels = [p[2] for p in info]
		cur_roleset_ids = [p[3] for p in info]
		cur_roleset_suffixes = [p[4] for p in info]
		cur_orig_toks = info[0][5]
		cur_v_idx = [p[6] for p in info]
		# (tokens, token length, vn labels, srl labels, orig tokens, v idx)
		rs.append((cur_toks, cur_tok_l, cur_vn_labels, cur_srl_labels, cur_roleset_ids, cur_roleset_suffixes, cur_orig_toks, cur_v_idx))

	print('writing grouped original tokens to {0}'.format(output + '.orig_tok_grouped.txt'))
	with open(output + '.orig_tok_grouped.txt', 'w') as f:
		for row in rs:
			f.write(row[6] + '\n')

	print('maximum number of predicates: {0}'.format(act_max_num_v))
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


def convert_semlink_pool(opt, vn_label_indexer, srl_label_indexer, roleset_indexer, semlink):
	semlink_map = np.ones((len(vn_label_indexer.d), len(roleset_indexer.d)), dtype=np.int32) * -1
	semlink_pool = []
	for (vc, roleset_id), pairs in tqdm(semlink.items(), desc="converting semlink pool"):
		vc = 'B-'+vc
		if vc not in vn_label_indexer.d or roleset_id not in roleset_indexer.d:
			continue
		vc_idx = vn_label_indexer.d[vc]
		roleset_idx = roleset_indexer.d[roleset_id]
		semlink_map[vc_idx, roleset_idx] = len(semlink_pool)
		pairs = pairs[:opt.max_num_semlink]
		role_idx = [srl_label_indexer.d['B-'+r] for r, _ in pairs]
		arg_idx = [vn_label_indexer.d['B-'+a] for _, a in pairs]
		semlink_pool.append(np.asarray([role_idx, arg_idx], dtype=np.int32))
	pool = np.zeros((len(semlink_pool), 2, opt.max_num_semlink), dtype=np.int32)
	for i, entry in enumerate(semlink_pool):
		pool[i, :, :entry.shape[1]] = entry

	valid_cnt = (semlink_map != -1).sum()
	print(f'pooled {valid_cnt} semlink entries.')
	return semlink_map, pool


def process(opt):
	tokenizer = get_tokenizer(opt.transformer_type)
	vn_label_indexer = Indexer(symbols=["O"], num_oov=0)	# label with O at the 0-th index
	srl_label_indexer = Indexer(symbols=["O"], num_oov=0)	# label with O at the 0-th index
	roleset_indexer = Indexer(symbols=["O"], num_oov=0)	# label with O at the 0-th index
	roleset_suffix_indexer = Indexer(symbols=["O"], num_oov=0)

	#### tokenize
	tokenizer_output = opt.tokenizer_output if opt.tokenizer_output != opt.dir else opt.dir
	train_toks, train_tok_l, train_vn_labels, train_srl_labels, train_roleset_ids, train_roleset_suffixes, train_orig_toks, train_v_ids = tokenize_and_write(opt, tokenizer, opt.train_vn, opt.train_srl, tokenizer_output + '.train')
	val_toks, val_tok_l, val_vn_labels, val_srl_labels, val_roleset_ids, val_roleset_suffixes, val_orig_toks, val_v_ids = tokenize_and_write(opt, tokenizer, opt.val_vn, opt.val_srl, tokenizer_output + '.val')
	test1_toks, test1_tok_l, test1_vn_labels, test1_srl_labels, test1_roleset_ids, test1_roleset_suffixes, test1_orig_toks, test1_v_ids = tokenize_and_write(opt, tokenizer, opt.test1_vn, opt.test1_srl, tokenizer_output + '.test1')
	if opt.test2_vn != opt.dir:
		test2_toks, test2_tok_l, test2_vn_labels, test2_srl_labels, test2_roleset_ids, test2_roleset_suffixes, test2_orig_toks, test2_v_ids = tokenize_and_write(opt, tokenizer, opt.test2_vn, opt.test2_srl, tokenizer_output + '.test2')

	print("First pass through data to get label vocab...")
	num_train_vn = make_label_vocab2(opt, vn_label_indexer, train_vn_labels, count=True)
	num_train_srl = make_label_vocab2(opt, srl_label_indexer, train_srl_labels, count=True)
	_ = make_label_vocab1(opt, roleset_indexer, train_roleset_ids, count=True)
	_ = make_label_vocab1(opt, roleset_suffix_indexer, train_roleset_suffixes, count=True)
	assert(num_train_vn == num_train_srl)
	print("Number of examples in training: {}".format(num_train_vn))

	num_val_vn = make_label_vocab2(opt, vn_label_indexer, val_vn_labels, count=True)
	num_val_srl = make_label_vocab2(opt, srl_label_indexer, val_srl_labels, count=True)
	_ = make_label_vocab1(opt, roleset_indexer, val_roleset_ids, count=True)
	_ = make_label_vocab1(opt, roleset_suffix_indexer, val_roleset_suffixes, count=True)
	assert(num_val_vn == num_val_srl)
	print("Number of examples in valid: {}".format(num_val_vn))

	num_test1_vn = make_label_vocab2(opt, vn_label_indexer, test1_vn_labels, count=False)	# no counting on test set
	num_test1_srl = make_label_vocab2(opt, srl_label_indexer, test1_srl_labels, count=False)
	_ = make_label_vocab1(opt, roleset_indexer, test1_roleset_ids, count=False)
	_ = make_label_vocab1(opt, roleset_suffix_indexer, test1_roleset_suffixes, count=True)
	assert(num_test1_vn == num_test1_srl)
	print("Number of examples in test1: {}".format(num_test1_vn))

	if opt.test2_vn != opt.dir:
		num_test2_vn = make_label_vocab2(opt, vn_label_indexer, test2_vn_labels, count=False)	# no counting on test set
		num_test2_srl = make_label_vocab2(opt, srl_label_indexer, test2_srl_labels, count=False)
		_ = make_label_vocab1(opt, roleset_indexer, test2_roleset_ids, count=False)
		_ = make_label_vocab1(opt, roleset_suffix_indexer, test2_roleset_suffixes, count=True)
		assert(num_test2_vn == num_test2_srl)
		print("Number of examples in test2: {}".format(num_test2_vn))

	vn_label_indexer.write(opt.output + ".vn_label.dict")
	srl_label_indexer.write(opt.output + ".srl_label.dict")
	roleset_indexer.write(opt.output + ".roleset_label.dict")
	roleset_suffix_indexer.write(opt.output + ".roleset_suffix.dict")
	print('vn label size: {}'.format(len(vn_label_indexer.d)))
	print('srl label size: {}'.format(len(srl_label_indexer.d)))
	print('roleset label size: {}'.format(len(roleset_indexer.d)))
	print('roleset suffix size: {}'.format(len(roleset_suffix_indexer.d)))

	semlink = load_semlink(opt.semlink)
	semlink_map, semlink_pool = convert_semlink_pool(opt, vn_label_indexer, srl_label_indexer, roleset_indexer, semlink)

	convert(opt, tokenizer, vn_label_indexer, srl_label_indexer, roleset_indexer, roleset_suffix_indexer,
		val_toks, val_tok_l, val_vn_labels, val_srl_labels, val_roleset_ids, val_roleset_suffixes, val_orig_toks, val_v_ids,
		semlink, semlink_map, semlink_pool,
		tokenizer_output+'.val', opt.output + ".val.hdf5")

	convert(opt, tokenizer, vn_label_indexer, srl_label_indexer, roleset_indexer, roleset_suffix_indexer,
		train_toks, train_tok_l, train_vn_labels, train_srl_labels, train_roleset_ids, train_roleset_suffixes, train_orig_toks, train_v_ids,
		semlink, semlink_map, semlink_pool,
		tokenizer_output+'.train', opt.output + ".train.hdf5")

	convert(opt, tokenizer, vn_label_indexer, srl_label_indexer, roleset_indexer, roleset_suffix_indexer,
		test1_toks, test1_tok_l, test1_vn_labels, test1_srl_labels, test1_roleset_ids, test1_roleset_suffixes, test1_orig_toks, test1_v_ids,
		semlink, semlink_map, semlink_pool,
		tokenizer_output+'.test1', opt.output + ".test1.hdf5")

	if opt.test2_vn != opt.dir:
		convert(opt, tokenizer, vn_label_indexer, srl_label_indexer, roleset_indexer, roleset_suffix_indexer,
			test2_toks, test2_tok_l, test2_vn_labels, test2_srl_labels, test2_roleset_ids, test2_roleset_suffixes, test2_orig_toks, test2_v_ids,
			semlink, semlink_map, semlink_pool,
			tokenizer_output+'.test2', opt.output + ".test2.hdf5")	
	
	
def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--train_vn', help="Path to training data, sentence and labels.", default = "upc_vn.train.vn.txt")
	parser.add_argument('--val_vn', help="Path to validation data, sentence and labels.",default = "upc_vn.val.vn.txt")
	parser.add_argument('--test1_vn', help="Path to test1 data, sentence and labels.",default = "upc_vn.test1.vn.txt")
	parser.add_argument('--test2_vn', help="Path to test2 data (optional).",default = "")
	parser.add_argument('--train_srl', help="Path to training data, sentence and labels.", default = "upc_vn.train.srl.txt")
	parser.add_argument('--val_srl', help="Path to validation data, sentence and labels.",default = "upc_vn.val.srl.txt")
	parser.add_argument('--test1_srl', help="Path to test1 data, sentence and labels.",default = "upc_vn.test1.srl.txt")
	parser.add_argument('--test2_srl', help="Path to test2 data (optional).",default = "")
	parser.add_argument('--dir', help="Path to the data dir",default = "./data/vn_modular/")
	parser.add_argument('--transformer_type', help="The type of transformer encoder from huggingface, eg. roberta-base",default = "roberta-base")
	
	parser.add_argument('--batch_size', help="Size of each minibatch.", type=int, default=8)
	parser.add_argument('--max_seq_l', help="Maximal sequence length", type=int, default=300)
	parser.add_argument('--max_num_v', help="Maximal number of predicate in a sentence", type=int, default=22)
	parser.add_argument('--max_num_subtok', help="Maximal number subtokens in a word", type=int, default=8)
	parser.add_argument('--max_num_semlink', help="Maximal number of semlink role-argument mapping for a predicate.", type=int, default=6)
	parser.add_argument('--tokenizer_output', help="Prefix of the tokenized output file names. ", type=str, default="upc_vn")
	parser.add_argument('--output', help="Prefix of the output file names. ", type=str, default="upc_vn")
	parser.add_argument('--shuffle', help="If = 1, shuffle sentences before sorting (based on source length).", type=int, default=1)
	parser.add_argument('--seed', help="The random seed", type=int, default=1)

	parser.add_argument('--semlink', help="Path to the semlink mapping", type=str, default="./semlink/instances/pb-vn2.json")
	opt = parser.parse_args(arguments)

	opt.train_vn = opt.dir + opt.train_vn
	opt.val_vn = opt.dir + opt.val_vn
	opt.test1_vn = opt.dir + opt.test1_vn
	opt.test2_vn = opt.dir + opt.test2_vn
	opt.train_srl = opt.dir + opt.train_srl
	opt.val_srl = opt.dir + opt.val_srl
	opt.test1_srl = opt.dir + opt.test1_srl
	opt.test2_srl = opt.dir + opt.test2_srl
	opt.output = opt.dir + opt.output
	opt.tokenizer_output = opt.dir + opt.tokenizer_output

	process(opt)

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
