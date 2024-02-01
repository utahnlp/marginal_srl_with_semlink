import sys
import argparse
import numpy as np
import h5py
from collections import defaultdict
from transformers import *
import json
from tqdm import tqdm
from . import extract
from util.util import Indexer, load_arg_role, parse_label
from .preprocess import make_label_vocab1, make_label_vocab2, load_arg_role, tokenize_and_write, load_semlink, extract_semlink, get_semlink_map, analyze_semlink, convert, get_tokenizer, analyze_arg_role_map, load_frameset, extract_frame_pool
import random
import functools


def manual_fix_for_semlink(semlink):
	semlink[('45.6', '01')].append(('ARG1', 'THEME'))

	return semlink


def corrupt_vnclass(toks, orig_toks, v_idx, gold, all_vnclass, probability):
	if random.uniform(0, 1) < probability:
		return random.choice(all_vnclass)
	else:
		return gold


def corrupt_sense(toks, orig_toks, v_idx, gold, all_senses, probability):
	if random.uniform(0, 1) < probability:
		# suffix = gold.split('.')[-1]
		# get variants for the same predicate
		# senses = [p for p in all_senses if p.split('.')[0] == gold.split('.')[0]]
		# if len(senses) != 1:
			# senses = [p for p in senses if p.split('.')[-1] != suffix]

		return random.choice(all_senses)
	else:
		return gold


def process(opt):
	tokenizer = get_tokenizer(opt.transformer_type)

	vn_label_indexer = Indexer.load_from(opt.indexer + '.vn_label.dict', oov_token='O')
	vn_class_indexer = Indexer.load_from(opt.indexer + '.vn_class.dict', oov_token='O')
	srl_label_indexer = Indexer.load_from(opt.indexer + '.srl_label.dict', oov_token='O')
	roleset_indexer = Indexer.load_from(opt.indexer + '.roleset_label.dict', oov_token='O')
	roleset_suffix_indexer = Indexer.load_from(opt.indexer + '.roleset_suffix.dict', oov_token='O')
	frame_indexer = Indexer.load_from(opt.indexer + '.frame.dict', oov_token='#')

	arg_role_map, arg_role_map_inv = load_arg_role(opt.arg_role_map)
	frameset = load_frameset(opt.frameset)

	#### tokenize
	val_vn_srl = path_prefix(opt.data, 'val') + '.vn_srl.txt'

	if opt.corrupt_type == 'vnclass':
		vnclass_model = functools.partial(corrupt_vnclass, all_vnclass=list(vn_class_indexer.d.keys()), probability=opt.probability)
	else:
		vnclass_model = None

	if opt.corrupt_type == 'sense':
		sense_model = functools.partial(corrupt_sense, all_senses=list(roleset_indexer.d.keys()), probability=opt.probability)
	else:
		sense_model = None

	val_output = path_prefix(opt.output, 'val')
	val_toks, val_tok_l, val_vn_labels, val_vn_classes, val_srl_labels, val_roleset_ids, val_roleset_suffixes, val_orig_toks, val_v_ids = tokenize_and_write(opt, vnclass_model, sense_model, tokenizer, val_vn_srl, val_output, unify_vn=False)

	print("Number of predicates in val: {}".format(len(val_toks)))

	if opt.aug_indexer == 1:
		print("First pass through data to get label vocab...")
		_ = make_label_vocab2(opt, vn_label_indexer, val_vn_labels, count=True)
		_ = make_label_vocab2(opt, srl_label_indexer, val_srl_labels, count=True)
		_ = make_label_vocab1(opt, vn_class_indexer, val_vn_classes, count=True)
		_ = make_label_vocab1(opt, roleset_indexer, val_roleset_ids, count=True)
		_ = make_label_vocab1(opt, roleset_suffix_indexer, val_roleset_suffixes, count=True)
		_ = make_label_vocab1(opt, frame_indexer, [p.split('.')[0] for p in val_roleset_ids], count=True)

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

	print('analyzing semlink on val set...')
	analyze_semlink(semlink_map, vn_class_indexer, roleset_suffix_indexer, vn_label_indexer, srl_label_indexer,
		val_vn_classes, val_roleset_suffixes, val_vn_labels, val_srl_labels)

	convert(opt, tokenizer, vn_label_indexer, vn_class_indexer, srl_label_indexer, roleset_indexer, roleset_suffix_indexer, frame_indexer, frame_pool,
		val_toks, val_tok_l, val_vn_labels, val_vn_classes, val_srl_labels, val_roleset_ids, val_roleset_suffixes, val_orig_toks, val_v_ids,
		semlink_map, val_output)


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
	parser.add_argument('--probability', help="The probability to corrupt a predicate label", type=float, default=0.05)
	parser.add_argument('--corrupt_type', help="The type of corruption, either vnclass or sense", default='vnclass')
	parser.add_argument('--indexer', help="Path to existing indexer dict (optional)", default="")
	parser.add_argument('--aug_indexer', help="Whether to augment indexer", type=int, default=1)
	parser.add_argument('--frameset', help="Path to extracted role set.", default = "frameset.txt")
	
	parser.add_argument('--batch_size', help="Size of each minibatch.", type=int, default=8)
	parser.add_argument('--max_seq_l', help="Maximal sequence length", type=int, default=200)
	parser.add_argument('--max_wp_l', help="Maximal length for predicate tokens to append to input", type=int, default=100)
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
