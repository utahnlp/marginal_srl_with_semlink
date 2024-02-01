import sys
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--unconstrained_output', default="./models/semi2_marginalsrl_iwcs_base_mergeconcat_semlink0_ep5_lr0000003_drop05_seed2_evalsemlink0_evalframe0")
parser.add_argument('--constrained_output', default="./models/semi2_marginalsrl_iwcs_base_mergeconcat_semlink0_ep5_lr0000003_drop05_seed2_evalsemlink1_evalframe0")


def load_quick_f1(path):
	f1_map = {}
	with open(path, 'r') as f:
		for line in tqdm(f, desc=f'path'):
			if line.strip() == '':
				continue
			parts = line.split('|||')
			f1 = float(parts[0].strip())
			ex_idx, v_idx = parts[1].strip().split(',')
			ex_idx = int(ex_idx)
			v_idx = int(v_idx)
			f1_map[(ex_idx, v_idx)] = f1
	return f1_map


def load_pretty_log(path):
	log_map = {}
	sent_map = {}
	with open(path, 'r') as f:
		blocks = f.read().split('\n\n')
		for block in tqdm(blocks, desc=f'path'):
			if block.strip() == '':
				continue
			lines = block.split('\n')
			parts = lines[0].split('|||')
			ex_idx = int(parts[0].strip())
			ex_idx = int(ex_idx)
			role_map = {}
			for k, pred in enumerate(lines[1].strip().split()):
				if pred != '-':
					role_map[k] = lines[2:][len(role_map)]

			log_map[ex_idx] = role_map
			sent_map[ex_idx] = parts[1].strip()
	return log_map, sent_map


def load_semlink(path, pretty_log_map):
	log_map = {}
	with open(path, 'r') as f:
		blocks = f.read().split('\n\n')
		for block in tqdm(blocks, desc=f'path'):
			if block.strip() == '':
				continue
			lines = block.split('\n')
			parts = lines[0].split('|||')
			ex_idx = int(parts[0].strip())
			ex_idx = int(ex_idx)
			constr_map = {}

			pretty_log = pretty_log_map[ex_idx]
			assert(len(pretty_log) == len(lines[1:]))

			for k, line in pretty_log.items():
				constr_map[k] = lines[1:][len(constr_map)]

			log_map[ex_idx] = constr_map
	return log_map


def load_frame(path, pretty_log_map):
	log_map = {}
	with open(path, 'r') as f:
		blocks = f.read().split('\n\n')
		for block in tqdm(blocks, desc=f'path'):
			if block.strip() == '':
				continue
			lines = block.split('\n')
			lines = block.split('\n')
			parts = lines[0].split('|||')
			ex_idx = int(parts[0].strip())
			ex_idx = int(ex_idx)
			frame_log = {}

			pretty_log = pretty_log_map[ex_idx]
			assert(len(pretty_log) == len(lines[1:]))

			for k, line in pretty_log.items():
				frame_log[k] = lines[1:][len(frame_log)]

			log_map[ex_idx] = frame_log

	return log_map


def load_all_log(opt, log_name):
	unconstrained_f1 = load_quick_f1(opt.unconstrained_output + f'.cross_crf_{log_name}_quick_f1.txt')
	unconstrained_pred, sent_map = load_pretty_log(opt.unconstrained_output + f'.cross_crf_{log_name}_pretty_pred.txt')
	unconstrained_gold, _ = load_pretty_log(opt.unconstrained_output + f'.cross_crf_{log_name}_pretty_gold.txt')
	assert(len(unconstrained_pred) == len(unconstrained_gold))

	constrained_f1 = load_quick_f1(opt.constrained_output + f'.cross_crf_{log_name}_quick_f1.txt')
	constrained_pred, _ = load_pretty_log(opt.constrained_output + f'.cross_crf_{log_name}_pretty_pred.txt')
	constrained_gold, _ = load_pretty_log(opt.constrained_output + f'.cross_crf_{log_name}_pretty_gold.txt')
	assert(len(constrained_pred) == len(constrained_gold))

	assert(len(unconstrained_f1) == len(constrained_f1))
	return unconstrained_f1, constrained_f1, unconstrained_pred, constrained_pred, sent_map, constrained_gold


def main(args):
	opt = parser.parse_args(args)

	uncon_vn_f1, con_vn_f1, uncon_vn_pred, con_vn_pred, sent_map, vn_gold = load_all_log(opt, 'vn')
	uncon_srl_f1, con_srl_f1, uncon_srl_pred, con_srl_pred, sent_map, srl_gold = load_all_log(opt, 'srl')

	semlink = load_semlink(opt.unconstrained_output + '.cross_crf_pretty_semlink.txt', con_vn_pred)
	frame = load_frame(opt.unconstrained_output + '.cross_crf_pretty_frame.txt', con_vn_pred)

	f1_cnt_map = {k: {'same': 0, 'worse': 0, 'better': 0} for k in ['vn', 'srl']}
	exact_cnt_map = {k: {'same': 0, 'diff': 0} for k in ['vn', 'srl']}
	for ex_v_idx in con_vn_f1.keys():
		ex_idx = ex_v_idx[0]
		v_idx = ex_v_idx[1]
		usrl_f1 = uncon_srl_f1[ex_v_idx]
		usrl_pred = uncon_srl_pred[ex_idx]
		srl_f1 = con_srl_f1[ex_v_idx]
		srl_pred = con_srl_pred[ex_idx]
		gsrl = srl_gold[ex_idx]
		constr = semlink[ex_idx]

		uvn_f1 = uncon_vn_f1[ex_v_idx]
		uvn_pred = uncon_vn_pred[ex_idx]
		vn_f1 = con_vn_f1[ex_v_idx]
		vn_pred = con_vn_pred[ex_idx]
		gvn = vn_gold[ex_idx]

		if usrl_f1 == srl_f1:
			f1_cnt_map['srl']['same'] += 1
		elif usrl_f1 < srl_f1:
			f1_cnt_map['srl']['better'] += 1
		elif usrl_f1 > srl_f1:
			f1_cnt_map['srl']['worse'] += 1

		if uvn_f1 == vn_f1:
			f1_cnt_map['vn']['same'] += 1
		elif uvn_f1 < vn_f1:
			f1_cnt_map['vn']['better'] += 1
		elif uvn_f1 > vn_f1:
			f1_cnt_map['vn']['worse'] += 1

		usrl_roles = usrl_pred[v_idx]
		srl_roles = srl_pred[v_idx]
		gsrl_roles = gsrl[v_idx]
		uvn_roles = uvn_pred[v_idx]
		vn_roles = vn_pred[v_idx]
		gvn_roles = gvn[v_idx]
		cur_constr = constr[v_idx]
		cur_frame = frame[ex_idx][v_idx]
		if srl_f1 < 1:
			print(f'>>>>>>>>>>>> {ex_idx} {v_idx}')
			print(sent_map[ex_idx])
			print('####### SRL')
			print('U', usrl_roles)
			print('C', srl_roles)
			for k in range(6):
				if srl_roles.count(f'(ARG{k}') > 1:
					print('unique role violation!')
			print('G', gsrl_roles)
			print('C f1', srl_f1)
			print('####### VN')
			print('U', uvn_roles)
			print('C', vn_roles)
			print('G', gvn_roles)
			print('C f1', vn_f1)
			print('####### Semlink')
			print(cur_constr)
			print('####### Frame')
			print(cur_frame)
			exact_cnt_map["srl"]['diff'] += 1
		else:
			exact_cnt_map["srl"]['same'] += 1

		if uvn_roles != vn_roles:
			exact_cnt_map["vn"]['diff'] += 1
		else:
			exact_cnt_map["vn"]['same'] += 1

	total_cnt = len(con_srl_f1)
	print(f'vn f1 same {f1_cnt_map["vn"]["same"]}/{total_cnt}')
	print(f'vn f1 worse {f1_cnt_map["vn"]["worse"]}/{total_cnt}')
	print(f'vn f1 better {f1_cnt_map["vn"]["better"]}/{total_cnt}')

	print(f'vn prediction same {exact_cnt_map["vn"]["same"]}')
	print(f'vn prediction diff {exact_cnt_map["vn"]["diff"]}')

	print(f'srl f1 same {f1_cnt_map["srl"]["same"]}/{total_cnt}')
	print(f'srl f1 worse {f1_cnt_map["srl"]["worse"]}/{total_cnt}')
	print(f'srl f1 better {f1_cnt_map["srl"]["better"]}/{total_cnt}')

	print(f'srl prediction same {exact_cnt_map["srl"]["same"]}')
	print(f'srl prediction diff {exact_cnt_map["srl"]["diff"]}')



if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))