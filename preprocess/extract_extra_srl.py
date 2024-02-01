import sys
import argparse
from tqdm import tqdm
from .extract import DEFAULT_VN_CLASS
from collections import defaultdict

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

def gen_vn_labels(opt, sent, srl_labels, v_idx):
	labels = ['O' for _ in srl_labels]
	labels[v_idx] = 'B-' + DEFAULT_VN_CLASS
	return labels

def fix_roleset(label):
	fix = lambda x: 'O' if x == '-1' else x
	return fix(label)

def fix_role(role):
	if '-ARG' in role:
		return role
	if '-AM' in role:
		return role.replace('-AM', '-ARGM')
	# change A0 to ARG0, and so.
	for i in range(10):	# 10 is more than enough
		if f'-A{i}' in role:
			return role.replace(f'-A{i}', f'-ARG{i}')
	return role

def load_filter(path):
	sent_v_map = defaultdict(list)
	with open(path, 'r') as f:
		for line in f:
			sent = line.split('|||')[0].split()
			v_idx = int(sent[0])
			sent = ' '.join(sent[1:])
			sent_v_map[sent].append(v_idx)
	print(f'loaded {len(sent_v_map)} sentences to filter.')
	return sent_v_map

def process(opt):
	all_v_idx, all_sent, all_vn, all_srl, all_sense = [], [], [], [], []

	filter_map = {}
	if opt.filter != opt.dir:
		filter_map = load_filter(opt.filter)

	filter_cnt = 0
	with open(opt.data, 'r') as f:
		for l in tqdm(f, desc="converting {0}".format(opt.data)):
			if l.strip() == '':
				continue

			parts = l.split('|||')
			assert(len(parts) == 4)

			sent, srl_labels, lemmas, sense = parts[0].strip().split(), parts[1].strip().split(), parts[2].strip().split(), parts[3].strip()
			v_idx = int(sent[0])
			sent = sent[1:]	# removing the first trigger idx
			sent = fix_toks(sent)
			assert(len(sent) == len(srl_labels))

			sent_str = ' '.join(sent)
			if sent_str in filter_map and v_idx in filter_map[sent_str]:
				filter_cnt += 1
				continue

			srl_labels = [fix_role(p) for p in srl_labels]
			sense = fix_roleset(sense)
			if sense != '-1':
				sense = f'{lemmas[v_idx]}.{sense}'
	
			vn_labels = gen_vn_labels(opt, sent, srl_labels, v_idx)
			all_vn.append(vn_labels)
			all_srl.append(srl_labels)
			all_sent.append(sent)
			all_v_idx.append(v_idx)
			all_sense.append(sense)

	print('filter_cnt', filter_cnt)
	print('output', opt.output)
	with open(opt.output, 'w') as f:
		for v_idx, sent, vn, srl, sense in tqdm(zip(all_v_idx, all_sent, all_vn, all_srl, all_sense)):
			f.write(f"{v_idx} {' '.join(sent)} ||| {' '.join(vn)} ||| {' '.join(srl)} ||| {sense}\n")

	
def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--data', help="Prefix to data name", default='../srl/conll05.train.txt')
	parser.add_argument('--filter', help="Prefix to data name", default='iwcs.train.vn_srl.txt')
	parser.add_argument('--dir', help="Path to the data dir",default = "./data/vn_modular/")
	parser.add_argument('--output', help="Prefix of the output file names. ", type=str, default="iwcs.extra.conll05.train.vn_srl.txt")

	opt = parser.parse_args(arguments)

	opt.data = opt.dir + opt.data
	opt.filter = opt.dir + opt.filter
	opt.output = opt.dir + opt.output

	process(opt)

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
