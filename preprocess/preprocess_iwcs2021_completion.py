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
from .preprocess import extract_semlink, load_semlink, manual_fix_for_semlink, get_tokenizer, pad, make_label_vocab2, make_label_vocab1, load_frameset, get_arg_mask, extract_frame_pool, sub2tok, unify_vn_label, vnclass_inf, sense_inf, tokenize_and_write, load_tokenized, load, analysis, get_semlink_map, analyze_arg_role_map, analyze_semlink


MOD_MAP = {
	'DIR': 'direction',
	'LOC': 'location',
	'EXT': 'extent',
	'REC': 'reciprocal',
	'PRD': 'secondary predicate',	# actually 2nd predication
	'PNC': 'purpose',
	'CAU': 'cause',
	'DIS': 'discourse',
	'ADV': 'adverb',
	'MOD': 'modal',
	'NEG': 'negation',
	'TMP': 'temporal',
	'MNR': 'manner',
	'TM': 'TM',	# seems like annotation error
	'ARGM': 'modifier',	# cases like ARGM but no suffix.
}


def map_pretty_arg(arg):
	rs = ''
	if arg == 'O':
		rs = arg
	if arg.startswith('B-') or arg.startswith('I-'):
		arg = arg[2:]
	if arg.startswith('R-') or arg.startswith('C-'):
		arg = arg[2:]

	if arg == 'V':
		rs = 'verb'
	if arg in (f'ARG{i}' for i in range(6)):
		rs = f'{arg[3]}'

	if arg == 'A' or arg == 'AA':
		rs = 'a'

	if arg.startswith('ARGM'):
		mod = arg.split('-')[-1]
		rs = f'{MOD_MAP[mod]}'

	if not rs:
		raise Exception(f'unrecognized {arg}')
	return rs



def convert(opt, tokenizer, vn_label_indexer, vn_class_indexer, srl_label_indexer, roleset_indexer, roleset_suffix_indexer, frame_indexer, frame_pool,
		toks, tok_l, vn_labels, vn_classes, srl_labels, roleset_ids, roleset_suffixes, orig_toks, v_ids,
		semlink_map, output):
	np.random.seed(opt.seed)

	act_max_wpa_l = 0
	max_num_v = 1

	inputs = tokenized_with_labels(opt, toks, tok_l, vn_labels, vn_classes, srl_labels, roleset_ids, roleset_suffixes, orig_toks, v_ids, output)
	num_v = len(inputs)

	tok_idx = np.zeros((num_v, opt.max_seq_l), dtype=np.int32)
	wpa_idx = np.zeros((num_v, opt.max_wpa_l), dtype=np.int32) + tokenizer.pad_token_id	# make it padding which will be auto filtered by transformer.
	sub2tok_idx = np.zeros((num_v, opt.max_seq_l, opt.max_num_subtok), dtype=np.int32) - 1	# empty is -1
	v_idx = np.zeros((num_v, max_num_v), dtype=np.int32)
	vn_label = np.zeros((num_v, max_num_v, opt.max_seq_l), dtype=np.int32)
	vn_class = np.zeros((num_v, max_num_v), dtype=np.int32)
	srl_label = np.zeros((num_v, max_num_v, opt.max_seq_l), dtype=np.int32)
	#
	frame_idx = np.zeros((num_v, max_num_v), dtype=int)
	#
	v_length = np.zeros((num_v, max_num_v), dtype=np.int32)	# number of v
	roleset_id = np.zeros((num_v, max_num_v), dtype=np.int32)	# just output, but deprecated
	roleset_suffix = np.zeros((num_v, max_num_v), dtype=np.int32)
	seq_length = np.zeros((num_v,), dtype=np.int32)
	orig_seq_length = np.zeros((num_v,), dtype=np.int32)
	ex_idx = np.zeros(num_v, dtype=np.int32)
	semlink = np.zeros((num_v, max_num_v, 2, opt.max_num_semlink), dtype=np.int32) - 1	# empty is -1
	semlink_l = np.zeros((num_v, max_num_v), dtype=np.int32)
	batch_keys = np.array([None for _ in range(num_v)])

	ex_id = 0
	num_prop = 0
	semlink_cnt = 0
	for _, (cur_toks, cur_tok_l, cur_vn_labels, cur_vn_class, cur_srl_labels, cur_roleset_id, cur_roleset_suffix, cur_orig_toks, cur_v_idx) in enumerate(tqdm(inputs, desc="converting inputs")):
		cur_toks = cur_toks.strip().split()
		cur_tok_l = [int(p) for p in cur_tok_l.strip().split()]
		cur_v_idx = int(cur_v_idx)

		tok_idx[ex_id, :len(cur_toks)] = np.array(tokenizer.convert_tokens_to_ids(cur_toks), dtype=int)
		v_length[ex_id] = 1

		# note that the resulted actual seq length after subtoken collapsing could be different even within the same batch
		#	actual seq length is the origial sequence length
		#	seq length is the length after subword tokenization
		cur_sub2tok = sub2tok(opt, cur_tok_l)
		cur_sub2tok = pad(cur_sub2tok, opt.max_seq_l, [-1 for _ in range(opt.max_num_subtok)])
		sub2tok_idx[ex_id] = np.array(cur_sub2tok, dtype=int)

		orig_seq_length[ex_id] = len(cur_tok_l)
		seq_length[ex_id] = len(cur_toks)
		batch_keys[ex_id] = seq_length[ex_id]

		cur_orig_toks = cur_orig_toks.strip().split()
		vn = cur_vn_labels.strip().split()
		srl = cur_srl_labels.strip().split()
		assert(len(vn) == len(srl) == len(cur_orig_toks))

		# put srl label
		wpa = [p for p in srl]
		wpa = [map_pretty_arg(p) for p in wpa]
		wpa[cur_v_idx] = cur_roleset_id
		wpa = tokenizer.tokenize(' '.join(wpa))
		act_max_wpa_l = max(act_max_wpa_l, len(wpa))
		wpa = wpa[:opt.max_wpa_l-1] + [tokenizer.sep_token]
		wpa_idx[ex_id, :len(wpa)] = np.array(tokenizer.convert_tokens_to_ids(wpa), dtype=int)

		v_idx[ex_id, 0] = cur_v_idx
		roleset_id[ex_id, 0] = roleset_indexer.convert(cur_roleset_id)

		vnc_id = vn_class_indexer.convert(cur_vn_class)
		vn_class[ex_id, 0] = vnc_id

		sense_id = roleset_suffix_indexer.convert(cur_roleset_suffix)
		roleset_suffix[ex_id, 0] = sense_id

		vn_id = vn_label_indexer.convert_sequence(vn)
		vn_label[ex_id, 0, :len(vn_id)] = np.array(vn_id, dtype=int)
		assert(vn_label[ex_id].sum() != 0)

		srl_id = srl_label_indexer.convert_sequence(srl)
		srl_label[ex_id, 0, :len(srl_id)] = np.array(srl_id, dtype=int)			
		assert(srl_label[ex_id].sum() != 0)

		lemma = cur_roleset_id.split('.')[0]
		frame_idx[ex_id, 0] = frame_indexer.d[lemma]

		semlink_alignments = semlink_map[vnc_id, sense_id]
		semlink_l[ex_id, 0] = (semlink_alignments[0] != -1).sum()
		semlink[ex_id, 0] = semlink_alignments
		if semlink_l[ex_id] != 0:
			semlink_cnt += 1

		ex_id += 1

	if opt.shuffle == 1:
		rand_idx = np.random.permutation(ex_id)
		tok_idx = tok_idx[rand_idx]
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
	print('act_max_wpa_l', act_max_wpa_l, 'cropped to', opt.max_wpa_l)
	analysis(tok_idx, tokenizer.unk_token_id)

	# Write output
	f = h5py.File(output + '.hdf5', "w")		
	f["tok_idx"] = tok_idx
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
	print("semlink was used {0} in {1} props".format(semlink_cnt, num_v))
	f.close()


def tokenized_with_labels(opt, toks, tok_l, vn_labels, vn_classes, srl_labels, roleset_ids, roleset_suffixes, orig_toks, v_ids, output):
	rs = []
	for cur_toks, cur_tok_l, cur_vn_labels, cur_vn_classes, cur_srl_labels, cur_roleset_id, cur_roleset_suffix, cur_orig_toks, cur_v_idx in zip(
		toks, tok_l, vn_labels, vn_classes, srl_labels, roleset_ids, roleset_suffixes, orig_toks, v_ids):
		rs.append((cur_toks, cur_tok_l, cur_vn_labels, cur_vn_classes, cur_srl_labels, cur_roleset_id, cur_roleset_suffix, cur_orig_toks, cur_v_idx))

	print('writing grouped original tokens to {0}'.format(output + '.orig_tok_grouped.txt'))
	with open(output + '.orig_tok_grouped.txt', 'w') as f:
		for row in rs:
			f.write(row[7] + '\n')
	return rs


def process(opt):
	tokenizer = get_tokenizer(opt.transformer_type)

	if opt.indexer != opt.dir:
		vn_label_indexer = Indexer.load_from(opt.indexer + '.vn_label.dict', oov_token='O')
		vn_class_indexer = Indexer.load_from(opt.indexer + '.vn_class.dict', oov_token='O')
		srl_label_indexer = Indexer.load_from(opt.indexer + '.srl_label.dict', oov_token='O')
		roleset_indexer = Indexer.load_from(opt.indexer + '.roleset_label.dict', oov_token='O')
		roleset_suffix_indexer = Indexer.load_from(opt.indexer + '.roleset_suffix.dict', oov_token='O')
		frame_indexer = Indexer.load_from(opt.indexer + '.frame.dict', oov_token='#')
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
	train_vn_srl = path_prefix(opt.data, 'train') + '.vn_srl.txt'
	val_vn_srl = path_prefix(opt.data, 'val') + '.vn_srl.txt'
	test1_vn_srl = path_prefix(opt.data, 'test1') + '.vn_srl.txt'
	test2_vn_srl = path_prefix(opt.data, 'test2') + '.vn_srl.txt'

	vnclass_model = None
	sense_model = None

	train_output = path_prefix(opt.output, 'train')
	train_toks, train_tok_l, train_vn_labels, train_vn_classes, train_srl_labels, train_roleset_ids, train_roleset_suffixes, train_orig_toks, train_v_ids = tokenize_and_write(opt, vnclass_model, sense_model, tokenizer, train_vn_srl, train_output, unify_vn=False)	# disable unify_vn since we are on semlink1.1

	print("Analyzing arg_role_map...")
	analyze_arg_role_map(arg_role_map, arg_role_map_inv, train_vn_labels, train_srl_labels)

	val_output = path_prefix(opt.output, 'val')
	val_toks, val_tok_l, val_vn_labels, val_vn_classes, val_srl_labels, val_roleset_ids, val_roleset_suffixes, val_orig_toks, val_v_ids = tokenize_and_write(opt, vnclass_model, sense_model, tokenizer, val_vn_srl, val_output, unify_vn=False)

	print("Analyzing arg_role_map...")
	analyze_arg_role_map(arg_role_map, arg_role_map_inv, val_vn_labels, val_srl_labels)
	
	test1_output = path_prefix(opt.output, 'test1')
	test1_toks, test1_tok_l, test1_vn_labels, test1_vn_classes, test1_srl_labels, test1_roleset_ids, test1_roleset_suffixes, test1_orig_toks, test1_v_ids = tokenize_and_write(opt, vnclass_model, sense_model, tokenizer, test1_vn_srl, test1_output, unify_vn=False)

	test2_output = path_prefix(opt.output, 'test2')
	test2_toks, test2_tok_l, test2_vn_labels, test2_vn_classes, test2_srl_labels, test2_roleset_ids, test2_roleset_suffixes, test2_orig_toks, test2_v_ids = tokenize_and_write(opt, vnclass_model, sense_model, tokenizer, test2_vn_srl, test2_output, unify_vn=False)

	print("Number of predicates in training: {}".format(len(train_toks)))
	print("Number of predicates in val: {}".format(len(val_toks)))
	print("Number of predicates in test1: {}".format(len(test1_toks)))
	print("Number of predicates in test2: {}".format(len(test2_toks)))

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

		_ = make_label_vocab2(opt, vn_label_indexer, test2_vn_labels, count=False)	# no counting on test set
		_ = make_label_vocab2(opt, srl_label_indexer, test2_srl_labels, count=False)
		_ = make_label_vocab1(opt, vn_class_indexer, test2_vn_classes, count=False)
		_ = make_label_vocab1(opt, roleset_indexer, test2_roleset_ids, count=False)
		_ = make_label_vocab1(opt, roleset_suffix_indexer, test2_roleset_suffixes, count=False)
		_ = make_label_vocab1(opt, frame_indexer, [p.split('.')[0] for p in test2_roleset_ids], count=False)

		# also record any extra suffix defined in frameset
		for _, p in frameset.items():
			for suffix, _ in p:
				roleset_suffix_indexer.register_all_words([suffix], count=False)

		indexer_output = f'{opt.output}'
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
	# semlink = manual_fix_for_semlink(semlink, collapse_vc=False)
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

	convert(opt, tokenizer, vn_label_indexer, vn_class_indexer, srl_label_indexer, roleset_indexer, roleset_suffix_indexer, frame_indexer, frame_pool,
		test2_toks, test2_tok_l, test2_vn_labels, test2_vn_classes, test2_srl_labels, test2_roleset_ids, test2_roleset_suffixes, test2_orig_toks, test2_v_ids,
		semlink_map, test2_output)


def path_prefix(data, split):
	return f'{data}.{split}'


def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--data', help="Prefix to data name", default='iwcs')
	parser.add_argument('--dir', help="Path to the data dir",default = "./data/vn_modular/")
	parser.add_argument('--transformer_type', help="The type of transformer encoder from huggingface, eg. roberta-base",default = "roberta-base")

	parser.add_argument('--arg_role_map', help="The path to valid arg-role mapping for semlink", default = "iwcs.arg_role.txt")

	parser.add_argument('--indexer', help="Path to existing indexer dict (optional)", default="")
	parser.add_argument('--aug_indexer', help="Whether to augment indexer", type=int, default=1)
	parser.add_argument('--frameset', help="Path to extracted role set.", default = "frameset.txt")
	
	parser.add_argument('--batch_size', help="Size of each minibatch.", type=int, default=8)
	parser.add_argument('--max_seq_l', help="Maximal sequence length", type=int, default=200)
	parser.add_argument('--max_wpa_l', help="Maximal length for aligned predicate tokens to append to input", type=int, default=200)
	parser.add_argument('--max_num_v', help="Maximal number of predicate in a sentence", type=int, default=21)
	parser.add_argument('--max_num_subtok', help="Maximal number subtokens in a word", type=int, default=8)
	parser.add_argument('--max_num_semlink', help="Maximal number of semlink role-argument mapping for a predicate.", type=int, default=24)
	parser.add_argument('--output', help="Prefix of the output file names. ", type=str, default="iwcs")
	parser.add_argument('--shuffle', help="If = 1, shuffle sentences before sorting (based on source length).", type=int, default=1)
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
