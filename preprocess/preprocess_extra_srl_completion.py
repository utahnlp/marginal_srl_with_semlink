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
from hf.v_model import VModel
from .preprocess import make_label_vocab1, make_label_vocab2, load_arg_role, tokenize_and_write, load_tokenized, load_semlink, extract_semlink, manual_fix_for_semlink, get_semlink_map, analyze_semlink, get_tokenizer, analyze_arg_role_map, load_frameset, extract_frame_pool
from .preprocess_iwcs2021_completion import convert

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

	vn_srl = opt.data
	output = opt.output

	vnclass_model = None
	sense_model = None

	toks, tok_l, vn_labels, vn_classes, srl_labels, roleset_ids, roleset_suffixes, orig_toks, v_ids = tokenize_and_write(opt, vnclass_model, sense_model, tokenizer, vn_srl, output)

	print("Analyzing arg_role_map...")
	analyze_arg_role_map(arg_role_map, arg_role_map_inv, vn_labels, srl_labels)

	print("Number of predicates in data: {}".format(len(toks)))

	if opt.aug_indexer == 1:
		print("First pass through data to get label vocab...")
		_ = make_label_vocab1(opt, frame_indexer, [lemma for lemma, _ in frameset.items()], count=False)
		_ = make_label_vocab2(opt, vn_label_indexer, vn_labels, count=True)
		_ = make_label_vocab2(opt, srl_label_indexer, srl_labels, count=True)
		_ = make_label_vocab1(opt, vn_class_indexer, vn_classes, count=True)
		_ = make_label_vocab1(opt, roleset_indexer, roleset_ids, count=True)
		_ = make_label_vocab1(opt, roleset_suffix_indexer, roleset_suffixes, count=True)
		_ = make_label_vocab1(opt, frame_indexer, [p.split('.')[0] for p in roleset_ids], count=True)

		# also record any extra suffix defined in frameset
		for _, p in frameset.items():
			for suffix, _ in p:
				roleset_suffix_indexer.register_all_words([suffix], count=False)

	frame_pool = extract_frame_pool(roleset_suffix_indexer, srl_label_indexer, frame_indexer, frameset)
	
	indexer_output = output
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

	sense_vn_map, suffix_vn_map = load_semlink(opt.semlink)
	semlink = extract_semlink(sense_vn_map, suffix_vn_map)
	semlink_map = get_semlink_map(opt, vn_label_indexer, vn_class_indexer, srl_label_indexer, roleset_suffix_indexer, semlink)

	print('analyzing semlink...')
	analyze_semlink(semlink_map, vn_class_indexer, roleset_suffix_indexer, vn_label_indexer, srl_label_indexer,
		vn_classes, roleset_suffixes, vn_labels, srl_labels)

	convert(opt, tokenizer, vn_label_indexer, vn_class_indexer, srl_label_indexer, roleset_indexer, roleset_suffix_indexer,
		frame_indexer, frame_pool,
		toks, tok_l, vn_labels, vn_classes, srl_labels, roleset_ids, roleset_suffixes, orig_toks, v_ids,
		semlink_map, output)

	
def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--data', help="Prefix to data name", default='iwcs.extra.conll05.train.vn_srl.txt')
	parser.add_argument('--dir', help="Path to the data dir",default = "./data/vn_modular/")
	parser.add_argument('--transformer_type', help="The type of transformer encoder from huggingface, eg. roberta-base",default = "roberta-base")

	parser.add_argument('--arg_role_map', help="The path to valid arg-role mapping for semlink", default = "iwcs.arg_role.txt")

	parser.add_argument('--indexer', help="Path to existing indexer dict (optional)", default="iwcs")
	parser.add_argument('--aug_indexer', help="Whether to augment indexer", type=int, default=1)
	parser.add_argument('--frameset', help="Path to extracted role set.", default = "frameset.txt")
	
	parser.add_argument('--batch_size', help="Size of each minibatch.", type=int, default=8)
	parser.add_argument('--max_seq_l', help="Maximal sequence length", type=int, default=200)
	parser.add_argument('--max_wpa_l', help="Maximal length for aligned predicate tokens to append to input", type=int, default=200)
	parser.add_argument('--max_num_v', help="Maximal number of predicate in a sentence", type=int, default=21)
	parser.add_argument('--max_num_subtok', help="Maximal number subtokens in a word", type=int, default=8)
	parser.add_argument('--max_num_semlink', help="Maximal number of semlink role-argument mapping for a predicate.", type=int, default=24)
	parser.add_argument('--output', help="Prefix of the output file names. ", type=str, default="iwcs.extra.conll05")
	parser.add_argument('--shuffle', help="If = 1, shuffle sentences before sorting (based on source length).", type=int, default=1)
	parser.add_argument('--seed', help="The random seed", type=int, default=1)

	parser.add_argument('--semlink', help="Path to the semlink json", default='iwcs')

	opt = parser.parse_args(arguments)

	opt.data = opt.dir + opt.data
	opt.output = opt.dir + opt.output
	opt.arg_role_map = opt.dir + opt.arg_role_map
	opt.frameset = opt.dir + opt.frameset
	opt.semlink = opt.dir + opt.semlink

	opt.indexer = opt.dir + opt.indexer
	

	process(opt)

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
