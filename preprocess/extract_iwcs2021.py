import sys
import argparse
from tqdm import tqdm
from xml.dom import minidom
import json
from collections import defaultdict
import torch
from util.util import parse_label, is_any_vnclass


PTB_MAPPINGS = {
    '-LRB-': '(',
    '-RRB-': ')',
    '-LCB-': '{',
    '-RCB-': '}',
    '-LSB-': '[',
    '-RSB-': ']',
    '\'\'': '\"',
    '``': '\"',
}

def fix_toks(sent):
	fixed = []
	for tok in sent:
		if tok in PTB_MAPPINGS:
			fixed.append(PTB_MAPPINGS[tok])
		else:
			fixed.append(tok)
	return fixed


def read_mappings_xml(mappings_xml):
    lemma_map = {}
    sense_vn_map = defaultdict(dict)	# roleset -> {vnclass -> list(role_arg_pairs)}
    suffix_vn_map = defaultdict(lambda: defaultdict(set))	# suffix -> {vncls -> set(role_arg_pairs)}
    for predicate in minidom.parse(mappings_xml).getElementsByTagName("predicate"):
        lemma = predicate.attributes['lemma'].value
        if lemma in lemma_map:
            raise ValueError('Repeat lemma found in mappings file {}: {}'.format(mappings_xml, lemma))
        roles = defaultdict(list)
        lemma_map[lemma] = roles

        for argmap in predicate.getElementsByTagName("argmap"):
            roleset = argmap.attributes['pb-roleset'].value.strip()
            vncls = argmap.attributes['vn-class'].value.strip()
            suffix = roleset.split('.')[-1]

            role_arg_pairs = []
            for role in argmap.getElementsByTagName("role"):
                pbarg = role.attributes['pb-arg'].value.strip()
                vntheta = role.attributes['vn-theta'].value.strip()
                if not vntheta or not pbarg:
                    print(f"Empty SemLink pb-arg to vn-arg mapping for {roleset} ARG{pbarg} {vntheta}")
                    continue
                role_arg_pairs.append((f'ARG{pbarg}'.upper(), vntheta.upper()))

            sense_vn_map[roleset].update({vncls: role_arg_pairs})
            suffix_vn_map[suffix][vncls].update(role_arg_pairs)

    print('extracted {0} sense_vn_map'.format(len(sense_vn_map)))
    print('extracted {0} suffix_vn_map'.format(len(suffix_vn_map)))
    return sense_vn_map, suffix_vn_map


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


def parse_joint_label(label):
	def fix_srl(p):
		return p.replace('-AM-', '-ARGM-')

	if '$' not in label:
		# for predicate, keep as-is
		if label in ['B-V', 'I-V']:
			return label, label
		# this is a SRL-only label
		return 'O', fix_srl(label)

	parts = label.strip().split('$')
	head = '-'.join(parts[0].split('-')[:-1])
	# make srl's AM -> ARGM
	srl_label = parts[0].split('-')[-1]
	srl_label = fix_srl(srl_label)
	srl_label = f'{head}-{srl_label}'
	# unify vn label
	vn_label = parts[1]
	vn_label = f'{head}-{vn_label}'
	return vn_label, srl_label

def extract_iwcs(path):
	all_v_ids = []
	all_sents = []
	all_vn_labels = []
	all_srl_labels = []
	all_senses = []

	skip_cnt = 0
	with open(path + '.txt', 'r') as f:
		for l_id, line in tqdm(enumerate(f), desc="parsing iwcs data"):
			if line.strip() == '':
				continue
			parts = line.split('|||')
			parts = [p.strip() for p in parts]
			assert(len(parts) == 5)

			toks = parts[0].split()
			v_idx = int(toks[0])
			toks = toks[1:]
			toks = fix_toks(toks)
			labels = parts[1].split()
			labels = [parse_joint_label(l) for l in labels]
			vn_labels = [l[0] for l in labels]
			srl_labels = [l[1] for l in labels]
			lemmas = parts[2].split()
			vnclass = parts[4]

			# A small set of examples in iwcs data has no vnclass.
			# TODO, should we include them anyways? wsj has a few dozon, brown has no.
			if vnclass.lower() == 'none':
				skip_cnt += 1
				continue

			assert(vn_labels[v_idx] == 'B-V')
			assert(srl_labels[v_idx] == 'B-V')

			# Minor adjustment to the v_idx to be consistent with lemma positions.
			# This is used in cases where there are B-V and I-V labels.
			# Eventually, only the token with lemma (the only lemma) is labeled as B-V, and I-V ignored.
			v_pos = [k for k, l in enumerate(srl_labels) if l in ('B-V', 'I-V')]
			actual_v_idx = [k for k in v_pos if lemmas[k] != '-']
			assert(len(actual_v_idx) == 1)
			v_idx = actual_v_idx[0]

			# wipe out B-V and I-V
			vn_labels = ['O' if l == 'I-V' or l == 'B-V' else l for l in vn_labels]
			srl_labels = ['O' if l == 'I-V' or l == 'B-V' else l for l in srl_labels]

			# vn labels use the vnclass for the predicate label
			vn_labels[v_idx] = f'B-{vnclass}'
			# also reset the srl B-V position
			srl_labels[v_idx] = 'B-V'


			roleset = f'{lemmas[v_idx]}.{parts[3]}'

			# finally, make uppercase labels
			vn_labels = [p.upper() for p in vn_labels]
			srl_labels = [p.upper() for p in srl_labels]

			all_v_ids.append(v_idx)
			all_sents.append(toks)
			all_vn_labels.append(vn_labels)
			all_srl_labels.append(srl_labels)
			all_senses.append(roleset)

	print(f'Skipped {skip_cnt} predicates due to incomplete/inconsistent format.')
	return all_v_ids, all_sents, all_vn_labels, all_srl_labels, all_senses



def write_extracted(all_v_ids, all_sents, all_vn_labels, all_srl_labels, all_senses, output):
	output = output + '.vn_srl.txt'
	print('writing to {0} examples to {1}...'.format(len(all_sents), output))
	with open(output, 'w') as f:
		for v_idx, sent, vn_labels, srl_labels, sense in tqdm(zip(all_v_ids, all_sents, all_vn_labels, all_srl_labels, all_senses),
			desc=f"writing output to {output}"):
			f.write('{0} {1} ||| {2} ||| {3} ||| {4}\n'.format(str(v_idx), ' '.join(sent), ' '.join(vn_labels), ' '.join(srl_labels), sense))


def extract_data(path, arg_role_map, output):
	all_v_ids, all_sents, all_vn_labels, all_srl_labels, all_senses = extract_iwcs(path)
	write_extracted(all_v_ids, all_sents, all_vn_labels, all_srl_labels, all_senses, output)
	return extract_arg_role_map(arg_role_map, all_vn_labels, all_srl_labels)


def expand_arg_role_mask(mask, size):
	rs = torch.zeros(size)
	rs[:mask.shape[0], :mask.shape[1]] = mask
	# also mark O-* and *-O as valid
	rs[0, :] = 1.0
	rs[:, 0] = 1.0
	return rs


def extract_arg_role_map(current_map, vn_labels, srl_labels):
	rows, cols, mask = current_map
	for vn, srl in zip(vn_labels, srl_labels):
		for v, s in zip(vn, srl):
			if is_any_vnclass(v) or s == 'B-V':
				continue
			v_head, v_body = parse_label(v)
			s_head, s_body = parse_label(s)
			if v_body not in rows:
				rows.append(v_body)
			if s_body not in cols:
				cols.append(s_body)

			if mask.shape != (len(rows), len(cols)):
				mask = expand_arg_role_mask(mask, (len(rows), len(cols)))
			mask[rows.index(v_body), cols.index(s_body)] = 1

	print(f'extracted arg_role_map with shape {mask.shape}.')
	return rows, cols, mask


def update_arg_role_map(current_map, suffix_vn_map):
	rows, cols, mask = current_map
	for _, entry in suffix_vn_map.items():
		for _, pairs in entry.items():
			for s, v in pairs:
				if v not in rows:
					rows.append(v)
				if s not in cols:
					cols.append(s)
	
				if mask.shape != (len(rows), len(cols)):
					mask = expand_arg_role_mask(mask, (len(rows), len(cols)))
				mask[rows.index(v), cols.index(s)] = 1

	print(f'updated arg_role_map with shape {mask.shape}.')
	return rows, cols, mask


def write_arg_role_map(current_map, output):
	print(f'writing arg_role_map to {output}...')
	rows, cols, mask = current_map
	with open(output, 'w') as f:
		f.write('NA' + '\t' + '\t'.join(cols) + '\n')
		for i, mask_row in enumerate(mask):
			f.write(rows[i] + '\t' + '\t'.join([str(p.item()) for p in mask_row.int()]) + '\n')


def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--dir', help="Path to the data dir", default="../verbnet-parsing-iwcs-2021_improved/semlink1.1/")
	parser.add_argument('--train', help="Path head to SRL training data.", default="train")
	parser.add_argument('--val', help="Path head to SRL validation txt.", default="dev")
	parser.add_argument('--test1', help="Path head to SRL test1 txt.", default="test1")
	parser.add_argument('--test2', help="Path head to SRL test2 txt.", default="test2")
	parser.add_argument('--semlink1_1', help="Path to semlink1.1 xml.", default="../verbnet-parsing-iwcs-2021_improved/semlink1.1/vn-pb/type_map.xml")
	#parser.add_argument('--guess', help="Level of guess when vn class annotation is missing, 0: no guess, 1: if only one in pb-vn then use it; 2: guess with best-match heuristic", type=int, default=0)
	parser.add_argument('--output', help="Prefix of the output file names. ", type=str, default = "./data/vn_modular/iwcs")
	opt = parser.parse_args(arguments)

	opt.train = opt.dir + opt.train
	opt.val = opt.dir + opt.val
	opt.test1 = opt.dir + opt.test1
	opt.test2 = opt.dir + opt.test2

	sense_vn_map, suffix_vn_map = read_mappings_xml(opt.semlink1_1)
	save_semlink(opt.output, sense_vn_map, suffix_vn_map)

	rows = ['O', 'V']
	cols = ['O', 'V']
	mask = torch.ones(2, 2)
	arg_role_map = (rows, cols, mask)
	arg_role_map = update_arg_role_map(arg_role_map, suffix_vn_map)

	arg_role_map = extract_data(opt.val, arg_role_map, opt.output + '.val')
	arg_role_map = extract_data(opt.train, arg_role_map, opt.output + '.train')
	arg_role_map = extract_data(opt.test1, arg_role_map, opt.output + '.test1')
	arg_role_map = extract_data(opt.test2, arg_role_map, opt.output + '.test2')

	write_arg_role_map(arg_role_map, opt.output + '.arg_role.txt')

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))