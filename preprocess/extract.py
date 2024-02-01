import sys
sys.path.append("./verbnet/")
import argparse
import json
from api.verbnet import *
import spacy
from spacy.tokenizer import Tokenizer
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import copy
from util.util import parse_label
from .bracket_parse import CustomizedBracketParseCorpusReader

DEFAULT_VN_CLASS = '26.4'	# just some place holder in default for mismatched sents (which only has SRL anyway)

stemmer = PorterStemmer()
spacy_nlp = spacy.load('en_core_web_sm')
spacy_nlp.tokenizer = Tokenizer(spacy_nlp.vocab)	# supposedly it tells spacy to only split on space

def fix_role(key):
	if key == 'NM':
		return None
	if key.startswith('CO-'):
		return 'CO_' + key[3:]
	return key

def fix_arg(key):
	key = key.upper()
	if key == 'NM' or key == 'NONE' or key == '':
		return None
	if key.startswith('CO-'):
		return 'CO_' + key[3:]
	return key

def collapse_vc_id(key):
	return key.split('-')[0]

# cleanup file names in epb
# 	only take the leaf file name
#	typically the names will end in .parse
#	in bolt, names end in .tree
# this function will simply rip them off
def cleanup_filename(n):
	n = n.split('/')[-1]	# ignore dirs

	# remove suffices
	# there would be other cases to handle, but there seems many different suffices.
	chunks = n.split('.')
	while len(chunks) > 0 and chunks[-1] in ['parse', 'prd', 'tree', 'xml', 'mrg', 'v10', 'v1', 'meta_removed']:
		chunks = chunks[:-1]
	
	return '.'.join(chunks)


def load_semlink_json(path, collapse_vc):
	print('loading semlink from {0} ...'.format(path))
	sense_vn_map = {}
	vn_sense_map = {}
	suffix_vn_map = {}
	all_args = set()	# just stats
	with open(path, 'r') as f:
		pb_vn = json.load(f)
		undef_cnt = 0
		for sense, vn_mapping in tqdm(pb_vn.items(), desc="loading pb-vn mapping from {0}".format(path)):
			if len(vn_mapping) == 0:
				undef_cnt += 1
				continue
			lemma = sense.split('.')[0]
			suffix = sense.split('.')[1]
			sense_vn_map[sense] = {}
			suffix_vn_map[suffix] = suffix_vn_map[suffix] if suffix in suffix_vn_map else {}

			# cluster 37.1-1 to 37.1
			if collapse_vc:
				collapsed_vn_mapping = {}
				for vc_id, mapping in vn_mapping.items():
					vc_id = collapse_vc_id(vc_id)
					if vc_id not in collapsed_vn_mapping:
						collapsed_vn_mapping[vc_id] = {}
					collapsed_vn_mapping[vc_id].update(mapping)
				vn_mapping = collapsed_vn_mapping

			for vc_id, mapping in vn_mapping.items():
				role_arg_pairs = set()
				for role, arg in mapping.items():
					arg = fix_arg(arg)
					role = fix_role(role)
					if arg is not None and arg != 'NONE' and role is not None and role != 'NONE':
						role_arg_pairs.add((role, arg))
				if vc_id not in vn_sense_map:
					vn_sense_map[vc_id] = set()
				vn_sense_map[vc_id].add(sense)
				sense_vn_map[sense].update({vc_id: role_arg_pairs})
				if vc_id not in suffix_vn_map[suffix]:
					suffix_vn_map[suffix][vc_id] = set()
				suffix_vn_map[suffix][vc_id].update(role_arg_pairs)

	print('{0}/{1} valid senses.'.format(len(pb_vn) - undef_cnt, len(pb_vn)))
	print('extracted {0} vn_sense_map and {1} sense_vn_map'.format(len(vn_sense_map), len(sense_vn_map)))
	print('extracted {0} suffix_vn_map'.format(len(suffix_vn_map)))
	return vn_sense_map, sense_vn_map, suffix_vn_map


def save_semlink(path, sense_vn_map, suffix_vn_map):
	print('writing to ', path + '.sense_vn.json')
	with open(path + '.sense_vn.json', 'w') as f:
		data = {}
		for sense, mapping in sense_vn_map.items():
			data[sense] = {k: list(p) for k, p in mapping.items()}
		f.write(json.dumps(data, indent=4, sort_keys=True))

	print('writing to ', path + '.suffix_vn.json')
	with open(path + '.suffix_vn.json', 'w') as f:
		data = {}
		for sense, mapping in suffix_vn_map.items():
			data[sense] = {k: list(p) for k, p in mapping.items()}
		f.write(json.dumps(data, indent=4, sort_keys=True))


def load_vn(vn_path, collapse_vc, use_coarse_vn=False):
	print('loading verbnet from {0} ...'.format(vn_path))
	vn = VerbNetParser(directory=vn_path)

	vn_map = {}
	vn_map_coarse = {}
	for vc in tqdm(vn.get_verb_classes(), desc="building verbnet and predicate sense mappings"):
		vc_id = '-'.join(vc.ID.split('-')[1:])	# rip off the lemma
		roles = set([role.role_type for role in vc.themroles])
	
		sub_vc_ids = []
		for sub_vc in vc.subclasses:
			sub_vc_id = '-'.join(sub_vc.ID.split('-')[1:])	# rip off the lemma
			sub_vc_ids.append(sub_vc_id)

			# also merge roles from sub vc to this vc
			if collapse_vc:
				sub_roles = set([role.role_type for role in sub_vc.themroles])
				roles.update(sub_roles)
	
		vn_map[vc_id] = {'roles': list(roles), 'sub_vc_ids': sub_vc_ids}
		if '-' not in vc_id:
			vn_map_coarse[vc_id] = vn_map[vc_id]
	
	print('{0} coarse vn classes and {1} fine-graned vn classes found.'.format(len(vn_map_coarse), len(vn_map)))

	if use_coarse_vn:
		vn_map = vn_map_coarse
		print('using coarse verbnet categorization')
	else:
		vn_map = vn_map
		print('using fine-grained verbnet categorization')

	return vn_map


def load_srl(path):
	rs = []
	lemmas = []
	with open(path, 'r') as f:
		rs = []
		line_cnt = 0
		for l in tqdm(f, desc="loading srl from {0}".format(path)):
			if l.strip() == '':	# typically this is the last line
				continue

			parts = l.strip().split('|||')
			assert(len(parts) == 4)		# predicate index, tokens, SRL, roleset suffix
			suffix = parts[-1].strip()

			sent, labels, lemmas = parts[0].strip().split(), parts[1].strip().split(), parts[2].strip().split()
			prop_id = int(sent[0])
			sent = sent[1:]
			v_idx = labels.index('B-V')
			if prop_id != v_idx:
				print('WARNING: propid {0} mismatches the index of B-V {1} at line {2}'.format(prop_id, v_idx, line_cnt))
			sense = f'{lemmas[v_idx]}.{suffix}'
			rs.append((v_idx, sent, labels, lemmas, sense))
			line_cnt += 1
	return rs


def load_file_id(path):
	rs = []
	with open(path, 'r') as f:
		for l in tqdm(f, desc="loading srl fileid from {0}".format(path)):
			if l.strip() == '':
				continue
			parts = l.strip().split('\t')
			assert(len(parts) == 2)
			rs.append((parts[0], int(parts[1])))	# file_id, sent_id
	print('loaded {0} lines'.format(len(rs)))
	return rs

def load_prop_gold(path):
	rs = []
	with open(path, 'r') as f:
		cur_block = []
		for l in tqdm(f, desc="loading srl prop gold from {0}".format(path)):
			if l.strip() == '':
				if len(cur_block) != 0:
					rs.append(cur_block)
					cur_block = []
				continue
			# actually only gonna take the lemma
			cur_block.append(l.strip().split('\t')[0])
	return rs

def _get_leaves(tree):
	rs = []
	for child in tree:
		if isinstance(child, str):
			rs.append((tree.label(), child))
		else:
			rs.extend(_get_leaves(child))

	return rs

def cut_leaves(leaves, label, sym):
	rs = []
	for l, t in leaves:
		if l != '-NONE-' and sym in t:
			sub_toks = t.split(sym)
			sub_toks = [p for pair in zip(sub_toks, [sym]*len(sub_toks)) for p in pair][:-1]
			sub_labels = [p for pair in zip([l]*len(sub_toks), [label]*len(sub_toks)) for p in pair][:-1]
			for sub_l, sub_t in zip(sub_labels, sub_toks):
				rs.append((sub_l, sub_t))
		else:
			rs.append((l, t))
	return rs

def get_leaves(tree):
	leaves = _get_leaves(tree)
	# further break down leaves if there is tokens being hyphened
	#leaves = cut_leaves(leaves, 'HYPH', '')
	#leaves = cut_leaves(leaves, 'HYPH', '-')
	#leaves = cut_leaves(leaves, 'SYM', '\\/')
	#leaves = cut_leaves(leaves, 'SYM', '/')
	return leaves


def load_treebank(path, file_suffix):
	reader = CustomizedBracketParseCorpusReader(path, f".*/.*\\.{file_suffix}")
	orig_file_ids = reader.fileids()
	#print('found {0} treebank files'.format(len(orig_file_ids)))

	rs = {}
	cnt = 0
	for file_id in tqdm(orig_file_ids, desc="loading treebank from {0}".format(path)):
		f = cleanup_filename(file_id)
		#f = file_id.split('/')[-1]
		#f = f.split('.mrg')[0]	# only take the leaf file name
		if (f, 0) in rs:
			#raise Exception('{0} already exists in parsed treebank'.format(f))
			print('{0} already exists in parsed treebank.'.format(f))

		for s_id, s in enumerate(reader.parsed_sents(file_id)):
			rs[(f, s_id)] = s
		cnt += 1

	return rs


def get_treebank_tok_ids(tree):
	leaves = get_leaves(tree)
	prop_v_map = []
	cur = 0
	for i, (l, t) in enumerate(leaves):
		if l.upper() == '-NONE-':
			prop_v_map.append(-1)
		else:
			prop_v_map.append(cur)
			cur += 1
	return prop_v_map

# load the mapping btw treebanks and vn rolesets from the semlink-2 EPB file
# the file has predicate sense annotated and may/may not have vn class annotated
#	in the later case, predicate sense can be combined with pb-to-vn mapping to get the actual vn roleset annotation
# the output is a dict with format (file_id, sent_id, prop_id) -> (lemma, vc_id, sense, explicit_role_arg_pairs)
def load_semlink_epb(path, treebank, collapse_vc):
	rs = {}
	missed_tb = set()
	with open(path, 'r') as f:
		for l in tqdm(f, desc="loading annotated semlink file from {0}".format(path)):
			if l.strip() == '':
				continue

			parts = l.strip().split(' ')
			file_id = cleanup_filename(parts[0])	# only take the file name
			sent_id = int(parts[1])
			prop_id = int(parts[2])
			lemma = parts[3]
			vc_id = parts[4]
			sense = parts[6]

			if vc_id == 'None':
				continue

			if collapse_vc:
				vc_id = collapse_vc_id(vc_id)

			if (file_id, sent_id) not in treebank:
				missed_tb.add(file_id)
				continue

			# explicit mapping
			#	there could be mappings that do not appear in pb-vn mapping, so better record them and use later
			tree = treebank[(file_id, sent_id)]
			explicit_role_arg_pairs = set()
			for part in parts[8:]:
				if '=' in part:
					subparts = part.split('=')
					assert(len(subparts) == 2)
					role = '-'.join(subparts[0].split('-')[1:])
					role = fix_role(role)
					arg = subparts[1].split(';')[0]
					arg = fix_arg(arg)
					explicit_role_arg_pairs.add((role, arg))

			# get the actual v index in terms of tokens
			#	since the prop_id in semlink points to the leaf node index in the parse tree
			tok_ids = get_treebank_tok_ids(tree)
			if prop_id >= len(tok_ids) or tok_ids[prop_id] == -1:
				print('WARNING: {0} has semlink pointed to the wrong token index'.format((file_id, sent_id, prop_id)))
				continue
			else:
				rs[(file_id, sent_id, tok_ids[prop_id])] = (lemma, vc_id, sense, explicit_role_arg_pairs)

	if len(missed_tb) > 0:
		print('skipped {0} semnlink annotations since no treebank found'.format(len(missed_tb)))
	return rs


def load_vn_ann(path, treebank, collapse_vc):
	rs = {}
	load_cnt = 0
	#subdirs = [f for f in listdir(path) if not isfile(join(path, f))]
	# only load from bolt, ewt, and google; ip-soft is left TODO
	subdirs = ['/bolt/', '/ewt/', '/google/']
	for sub in subdirs:
		sub = path + sub
		ann_files = [f for f in listdir(sub) if isfile(join(sub, f)) and f.endswith('.ann')]
		for f in tqdm(ann_files, desc="loading vn annotation from {0}".format(sub)):
			f_rs = load_vn_ann_file(join(sub, f), treebank, collapse_vc)
			if len(f_rs) != 0:
				load_cnt += 1
			rs.update(f_rs)
	#print('loaded {0} vn annotations from {1} files'.format(len(rs), load_cnt))
	return rs

# load the mapping btw treebanks and vn rolesets from .ann files
# the .ann files has vn class id directly annotated
#	and the file name contains predicate sense
# we gonna track both for a predicate
#	the output dict has format (file_id, sent_id, prop_id) = (lemma, vc_id, sense, explicit_role_arg_pairs={})
def load_vn_ann_file(path, treebank, collapse_vc):
	def get_lemma_and_sense_from_path(path):
		assert(path.endswith('.ann'))
		file_name = path[:-4].split('/')[-1]	# only take the last path chunk
		if '-' in file_name:
			parts = file_name.split('-')
			if parts[1].isnumeric():
				return parts[0], parts[1]
			else:
				parts[0], None
		elif '.' in file_name:
			parts = file_name.split('.')
			if parts[1].isnumeric():
				return parts[0], parts[1]
			else:
				parts[0], None

		return None, None

	# for .ann files, the sense id is in the file name
	f_lemma, suffix = get_lemma_and_sense_from_path(path)
	if f_lemma is None or suffix is None:
		return []

	rs = {}
	missed_tb = set()
	with open(path, 'r') as f:
		for l in f:
			if l.strip() == '':
				continue

			parts = l.strip().split(' ')
			file_id = cleanup_filename(parts[0])	# only take the file name
			sent_id = int(parts[1])
			prop_id = int(parts[2])
			lemma = parts[3]
			vc_id = parts[4]
			if '-' in vc_id:
				vc_id = '-'.join(vc_id.split('-')[1:])	# only take the ids without lemma

			# special treatment on lemma
			#	ipsoft-clean part has lemma-v but we only need lemma
			if lemma.endswith('-v'):
				lemma = lemma[:-2]

			# sanity check if lemma is the same as f_lemma
			if lemma != f_lemma:
				print('WARNING: lemma mismatch {0} vs {1} in {2}'.format(lemma, f_lemma, path))
			sense = f'{lemma}.{suffix}'

			if vc_id == 'None':
				continue

			if collapse_vc:
				vc_id = collapse_vc_id(vc_id)

			if (file_id, sent_id) not in treebank:
				missed_tb.add(file_id)
				continue

			# get the actual v index in terms of tokens
			#	since the prop_id in semlink points to the leaf node index in the parse tree
			tree = treebank[(file_id, sent_id)]
			tok_ids = get_treebank_tok_ids(tree)
			explicit_role_arg_pairs = {}
			if prop_id >= len(tok_ids) or tok_ids[prop_id] == -1:
				print('WARNING: {0} has semlink pointed to the wrong token index'.format((file_id, sent_id, prop_id)))
				continue
			else:
				rs[(file_id, sent_id, tok_ids[prop_id])] = (lemma, vc_id, sense, explicit_role_arg_pairs)

	#if len(missed_tb) > 0:
	#	print('skipped {0} vn annotations since no treebank found'.format(len(missed_tb)))
	return rs


def extend_semlink_from_epb(sense_vn_map, suffix_vn_map, epb):
	for (file_id, sent_id, prop_id), (lemma, vc_id, sense, explicit_role_arg_pairs) in epb.items():
		if sense not in sense_vn_map:
			sense_vn_map[sense] = {}
		if vc_id not in sense_vn_map[sense]:
			sense_vn_map[sense][vc_id] = set()
		sense_vn_map[sense][vc_id].update(explicit_role_arg_pairs)

		suffix = sense.split('.')[-1]
		if suffix not in suffix_vn_map:
			suffix_vn_map[suffix] = {}
		if vc_id not in suffix_vn_map[suffix]:
			suffix_vn_map[suffix][vc_id] = set()
		suffix_vn_map[suffix][vc_id].update(explicit_role_arg_pairs)
	return sense_vn_map, suffix_vn_map


def extract_from_srl(data, semlink_epb, sense_vn_map):
	srl = load_srl(data + '.txt')
	file_ids = load_file_id(data + '.fileid.txt')
	orig_file_ids = load_file_id(data + '.origfileid.txt')
	prop_golds = load_prop_gold(data + '.props.gold.txt')

	print('Extracting verbnet labels from srl...')
	#line_cnt = 0
	not_in_semlink_cnt = 0

	ambiguous_arg_cnt = 0
	matched_arg_cnt = 0
	matched = {}
	mismatched = {}
	for line_id, (srl_entry, file_id, orig_file_id) in enumerate(zip(srl, file_ids, orig_file_ids)):
		v_idx, sent, srl_labels, lemmas, sense = srl_entry
		file_id, sent_id = file_id
		file_id = cleanup_filename(file_id)
		orig_file_id, _ = orig_file_id

		# if this is not in semlink, it's srl only.
		if (file_id, sent_id, v_idx) not in semlink_epb:
			vn_labels = ['O' for _ in srl_labels]
			vn_labels[v_idx] = 'B-' + DEFAULT_VN_CLASS
			mismatched[(file_id, sent_id, v_idx)] = (sent, vn_labels, sense, srl_labels)
			continue

		(lemma, vc_id, sense_found, explicit_role_arg_pairs) = semlink_epb[(file_id, sent_id, v_idx)]
		if sense != sense_found:
			print('WARNING: {0} forcing {1} to {2} found in semlink_epb'.format((file_id, sent_id, v_idx), sense, sense_found))
			sense = sense_found

		if sense in sense_vn_map:
			if vc_id not in sense_vn_map[sense]:
				raise Exception((file_id, sent_id, v_idx), 'sense {0} with vn class {1} defined in semlink but not found in sense_vn_map.'.format(sense, vc_id))

			role_arg_pairs = sense_vn_map[sense][vc_id]
			vn_labels = []
			for l in srl_labels:
				if l == 'O':
					vn_labels.append('O')
					continue
				prefix, role = parse_label(l)
				args_found = list(a for (r, a) in role_arg_pairs if r == role)
				# TODO, what do we do here can make it better?
				if not args_found or len(args_found) > 1:
					vn_labels.append('O')
					if l.startswith('B-') and len(args_found) > 1:
						ambiguous_arg_cnt += 1
					continue
				if l.startswith('B-'):
					matched_arg_cnt += 1
				vn_labels.append(prefix + args_found[0])

			assert(len(vn_labels) == len(srl_labels))

			# force the predicate to have proper vn class id
			vn_labels[v_idx] = 'B-' + vc_id
			matched[(file_id, sent_id, v_idx)] = (sent, vn_labels, sense, srl_labels)
		else:
			print('{0} has no vn class {1} associated with sense {2}'.format((file_id, sent_id), vc_id, sense))

	print('{0} ambiguous, {1} matched vn args'.format(ambiguous_arg_cnt, matched_arg_cnt))
	print('{0}/{1} vn-srl examples extracted'.format(len(matched), len(srl)))
	print('{0}/{1} srl-only examples extracted'.format(len(mismatched), len(srl)))
	return matched, mismatched



def write_extracted(rs, output):
	vn_path = output + '.vn_srl.txt'
	print('writing to {0} examples to {1}...'.format(len(rs), vn_path))
	with open(vn_path, 'w') as f:
		for (file_id, sent_id, v_idx), (sent, vn_labels, sense, srl_labels) in rs.items():
			f.write('{0} {1} ||| {2} ||| {3} ||| {4}\n'.format(str(v_idx), ' '.join(sent), ' '.join(vn_labels), ' '.join(srl_labels), sense))

	fileid_path = output + '.fileid.txt'
	with open(fileid_path, 'w') as f:
		for (file_id, sent_id, v_idx), (sent, vn_labels, sense, srl_labels) in rs.items():
			f.write('{0} {1}\n'.format(file_id, sent_id))


def extract_data(srl_path, semlink_epb, sense_vn_map, output):
	matched, mismatched = extract_from_srl(srl_path, semlink_epb, sense_vn_map)
	write_extracted(matched, output + '.matched')
	write_extracted(mismatched, output + '.mismatched')


def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--verbnet', help="Path to verbnet directory that contains xml files.", default="./verbnet/verbnet3.4/")
	parser.add_argument('--pb_vn', help="Path to semlink propbank to verbnet mapping json file.",default="./semlink/instances/pb-vn2.json")
	parser.add_argument('--wsj_semlink', help="Path to semlink for WSJ data, a EPB-formatted txt file.",default="./semlink/instances/semlink-2")
	parser.add_argument('--treebank_3', help="Path to the treebank_3 directory that has parse files.",default="./data/treebank_3/parsed/prd/")
	parser.add_argument('--combined_treebank', help="Path to the combined treebank directory that has parse files. (This dir is specifically shared by Ghazaleh for updated treebanks)",default="./data/combined_treebank/")
	parser.add_argument('--vn_ann', help="Path to vn-annotation dir.", default="./vn-annotation/")
	parser.add_argument('--dir', help="Path to the data dir", default="./data/")
	parser.add_argument('--train', help="Path head to SRL training data.", default="unified-propbank-conll/all_train/upc_all_train")
	parser.add_argument('--val', help="Path head to SRL validation txt.", default="unified-propbank-conll/all_dev/upc_all_dev")
	parser.add_argument('--test1', help="Path head to SRL test1 txt.", default="unified-propbank-conll/all_test/upc_all_test")
	parser.add_argument('--test2', help="Path head to SRL test2 txt (optional).", default="")
	parser.add_argument('--collapse_vc', help="Whether collapse vn class id.", type=int, default=0)
	#parser.add_argument('--guess', help="Level of guess when vn class annotation is missing, 0: no guess, 1: if only one in pb-vn then use it; 2: guess with best-match heuristic", type=int, default=0)
	parser.add_argument('--output', help="Prefix of the output file names. ", type=str, default = "vn_modular/upc_vn")
	opt = parser.parse_args(arguments)

	opt.train = opt.dir + opt.train
	opt.val = opt.dir + opt.val
	opt.test1 = opt.dir + opt.test1
	opt.test2 = opt.dir + opt.test2
	opt.output = opt.dir + opt.output

	vn_map = load_vn(opt.verbnet, opt.collapse_vc)
	vn_sense_map, sense_vn_map, suffix_vn_map = load_semlink_json(opt.pb_vn, opt.collapse_vc)

	missed_vc_cnt = 0
	for vc_id, _ in vn_map.items():
		if vc_id not in vn_sense_map:
			missed_vc_cnt += 1
	print('{0}/{1} valid vn classes'.format(len(vn_map) - missed_vc_cnt, len(vn_map)))

	# load all treebanks
	treebank = {}
	# treebank_3 first
	print('Loading from treebank_3...')
	treebank.update(load_treebank(opt.treebank_3 + '/wsj/', file_suffix='prd'))
	treebank.update(load_treebank(opt.treebank_3 + '/brown/', file_suffix='prd'))
	# then, combined_treebank
	print('Loading from combined_treebank...')
	treebank.update(load_treebank(opt.combined_treebank + '/wsj/', file_suffix='parse'))
	treebank.update(load_treebank(opt.combined_treebank + '/bolt/SMS/', file_suffix='parse'))
	treebank.update(load_treebank(opt.combined_treebank + '/bolt/DF/', file_suffix='parse'))
	treebank.update(load_treebank(opt.combined_treebank + '/bolt/CTS/', file_suffix='parse'))
	treebank.update(load_treebank(opt.combined_treebank + '/google/EnglishWebTreebank/', file_suffix='parse'))

	with open('./all_treebanks.txt', 'w+') as f:
		for k in treebank.keys():
			f.write(f'{k[0]}\t{k[1]}\n')

	# load all vn-prop_sense links
	semlink_epb = {}
	semlink_epb.update(load_semlink_epb(opt.wsj_semlink, treebank, opt.collapse_vc))
	semlink_epb.update(load_vn_ann(opt.vn_ann, treebank, opt.collapse_vc))

	#semlink = filter_semlink_by_treebank(semlink, treebank)
	# extend the mapping from predicate sense to vn class from the wsj data
	sense_vn_map, suffix_vn_map = extend_semlink_from_epb(sense_vn_map, suffix_vn_map, semlink_epb)

	extract_data(opt.val, semlink_epb, sense_vn_map, opt.output + '.val')
	extract_data(opt.train, semlink_epb, sense_vn_map, opt.output + '.train')
	extract_data(opt.test1, semlink_epb, sense_vn_map, opt.output + '.test1')
	if opt.test2 != opt.dir:
		extract_data(opt.test2, semlink, sense_vn_map, opt.output + '.test2')

	save_semlink(opt.output, sense_vn_map, suffix_vn_map)

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))