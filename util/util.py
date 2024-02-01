from os import path
import h5py
import torch
from torch import nn
from torch import cuda
from collections import Counter
import numpy as np

class Indexer:
	def __init__(self, symbols = ["<blank>"], oov_token="<blank>"):
		self.oov_token = oov_token
		self.d = {}
		self.inv = {}
		self.cnt = {}
		for s in symbols:
			self.d[s] = len(self.d)
			self.cnt[s] = 0


	@classmethod
	def load_from(cls, path, oov_token="<blank>"):
		indexer = Indexer(symbols=[], oov_token=None)
		indexer.d = {}
		indexer.cnt = {}

		with open(path, 'r') as f:
			for line in f:
				if line.strip() == '':
					continue
				tok, idx, freq = line.split(' ')
				indexer.d[tok] = int(idx)
				indexer.cnt[tok] = int(freq)
				indexer.inv[idx] = tok
		indexer.oov_token = oov_token
		return indexer
			
	def convert(self, w):
		return self.d[w] if w in self.d else self.d[self.oov_token]

	def convert_sequence(self, ls):
		return [self.convert(l) for l in ls]

	def write(self, outfile, with_cnt=True):
		assert(len(self.d) == len(self.cnt))
		with open(outfile, 'w+') as f:
			items = [(v, k) for k, v in self.d.items()]
			items.sort()
			for v, k in items:
				if with_cnt:
					f.write('{0} {1} {2}\n'.format(k, v, self.cnt[k]))
				else:
					f.write('{0} {1}\n'.format(k, v))

	#   NOTE, only do counting on training set
	def register_all_words(self, seq, count):
		for w in seq:
			if w not in self.d:
				self.d[w] = len(self.d)
				self.cnt[w] = 0
				self.inv[self.d[w]] = w
			if w in self.cnt:
				self.cnt[w] = self.cnt[w] + 1 if count else self.cnt[w]


def compact_subtoks(enc, sub2tok_idx, compact_mode):
	batch_l, source_l, hidden_size = enc.shape
	if compact_mode == 'whole_word':
		enc = batch_index2_select(enc, sub2tok_idx, nul_idx=-1)
		return enc.sum(2)	# (batch_l, seq_l, hidden_size)
	elif compact_mode == 'first_subtok':
		enc = batch_index2_select(enc, sub2tok_idx[:, :, :1], nul_idx=-1)
		return enc.squeeze(2)	# (batch_l, seq_l, hidden_size)
	else:
		raise Exception('unrecognized compact_mode: {}'.format(compact_mode))


def random_interleave(ls1, ls2):
	which_ls = [0] * len(ls1) + [1] * len(ls2)
	which_ls = [which_ls[k] for k in torch.randperm(len(which_ls))]
	ls1 = ls1[::-1]
	ls2 = ls2[::-1]
	rs = []
	for k in which_ls:
		if k == 0:
			rs.append(ls1.pop())
		else:
			rs.append(ls2.pop())
	return rs


def load_arg_role(path):
	args, roles = [], []
	arg_role = []
	arg_role_map_inv = {}
	with open(path, 'r') as f:
		line_id = 0
		for line in f:
			if line.strip() == '':
				line_id += 1
				continue

			parts = line.strip().split()
			if line_id == 0:
				roles = parts[1:]
			else:
				args.append(parts[0])
				assert(len(parts[1:]) == len(roles))
				for p, r in zip(parts[1:], roles):
					if p == '1':
						arg_role.append((args[-1], r))
						arg_role_map_inv[len(arg_role)-1] = (args[-1], r)

			line_id += 1
	return arg_role, arg_role_map_inv


def parse_label(label):
	special_prefix = ['B-C-', 'B-R-', 'I-C-', 'I-R-']
	if label[:4] in special_prefix:
		return label[:4], label[4:]
	elif label[:2] in ['B-', 'I-']:
		return label[:2], label[2:]
	return None, label


def is_any_vnclass(label):
	# this is more strict than is_v since it also covers, e.g., B-C-109-1-1.
	return label.split('-')[-1][0].isnumeric()


def complete_opt(opt):
	if 'base' in opt.bert_type:
		opt.hidden_size = 768
	elif 'large' in opt.bert_type:
		opt.hidden_size = 1024

	if hasattr(opt, 'vn_label_dict'):
		if path.exists(opt.vn_label_dict):
			print('loading label dict from', opt.vn_label_dict)
			opt.vn_labels, opt.vn_label_map_inv = load_label_dict(opt.vn_label_dict)
		else:
			raise Exception('vn_label_dict does not exists', opt.vn_label_dict)

	if hasattr(opt, 'srl_label_dict'):
		if path.exists(opt.srl_label_dict):
			print('loading label dict from', opt.srl_label_dict)
			opt.srl_labels, opt.srl_label_map_inv = load_label_dict(opt.srl_label_dict)
		else:
			raise Exception('srl_label_dict does not exists', opt.srl_label_dict)

	if hasattr(opt, 'vn_class_dict'):
		if path.exists(opt.vn_class_dict):
			print('loading vn class dict from', opt.vn_class_dict)
			opt.vn_classes, opt.vn_class_map_inv = load_label_dict(opt.vn_class_dict)
			opt.num_vn_class = len(opt.vn_classes)

	if hasattr(opt, 'vn_label_map_inv'):
		opt.vn_label_map_inv = {int(k): v for k, v in opt.vn_label_map_inv.items()}
	if hasattr(opt, 'vn_labels'):
		opt.num_vn_label = len(opt.vn_labels)

	if hasattr(opt, 'srl_label_map_inv'):
		opt.srl_label_map_inv = {int(k): v for k, v in opt.srl_label_map_inv.items()}
	if hasattr(opt, 'srl_labels'):
		opt.num_srl_label = len(opt.srl_labels)

	if hasattr(opt, 'roleset_dict'):
		opt.roleset, opt.roleset_map_inv = load_label_dict(opt.roleset_dict)

	if hasattr(opt, 'roleset_suffix_dict'):
		opt.roleset_suffix, opt.roleset_suffix_map_inv = load_label_dict(opt.roleset_suffix_dict)

	if hasattr(opt, 'vn_labels'):
		opt.v_mask = get_v_mask(opt.vn_labels)
		opt.v_mask[0] = True	# enable O
		opt.v_idx = [i for i, m in enumerate(opt.v_mask) if m]
		opt.num_v_label = len(opt.v_idx)

	if hasattr(opt, 'arg_role_map'):
		opt.arg_role, opt.arg_role_map_inv = load_arg_role(opt.arg_role_map)

	if hasattr(opt, 'vn_labels') and hasattr(opt, 'srl_labels'):
		opt.condensed_labels = []
		opt.condensed_map_inv = {}
		opt.cross2condensed = [[-1 for _ in opt.srl_labels] for _ in opt.vn_labels]	# a matrix of (num_vn_label, num_srl_label) with indices to condensed labels
		opt.condensed2cross = []
		opt.condensed2vn = []	# vn label indices in the condensed list
		opt.condensed2srl = []	# srl label indices in the condensed list
		for i, v in enumerate(opt.vn_labels):
			v_prefix, v_root = parse_label(v)
			for k, s in enumerate(opt.srl_labels):
				s_prefix, s_root = parse_label(s)

				# skip B-x aligning to I-x
				if v_prefix is not None and s_prefix is not None and v_prefix != s_prefix:
					continue
				# skip verb aligning to non-verb
				if is_v(v) != is_v(s):
					continue
				# lastly check the arg-role mapping, if there is any
				if hasattr(opt, 'arg_role') and (v_root, s_root) not in opt.arg_role:
					continue
				# add this label
				prefix = v_prefix if v_prefix else s_prefix
				if not prefix:
					cross_label = 'O'
				else:
					cross_label = prefix + f'{v_root}/{s_root}'
				opt.condensed_labels.append(cross_label)
				opt.condensed_map_inv[len(opt.condensed_labels)-1] = cross_label
				opt.cross2condensed[i][k] = len(opt.condensed_labels)-1
				opt.condensed2cross.append((i, k))
				opt.condensed2vn.append(i)
				opt.condensed2srl.append(k)
		print('extracted {0} cross labels'.format(len(opt.condensed_labels)))

	return opt

def get_v_mask(labels):
	return [is_v(l) for l in labels]

def is_v(l):
	return l.startswith('B-') and (l.split('-')[1][0].isnumeric() or l.split('-')[1] == 'V')

def len_to_mask(lengths, max_len=-1):
	# assumes lengths to have shape (batch_size,)
	max_len = lengths.max() if max_len == -1 else max_len
	assert(max_len >= 0)
	batch_size = lengths.shape[0]
	pos = torch.arange(max_len).to(lengths.device).view(1, -1).expand(batch_size, -1)
	lengths = lengths.view(batch_size, 1).expand(batch_size, max_len)
	return lengths > pos 

def load_label_dict(label_dict):
	labels = []
	label_map_inv = {}
	with open(label_dict, 'r') as f:
		for l in f:
			if l.strip() == '':
				continue
			toks = l.rstrip().split()
			labels.append(toks[0])
			label_map_inv[int(toks[1])] = toks[0]
	return labels, label_map_inv

def get_special_tokens(tokenizer):
	CLS, SEP = tokenizer.cls_token, tokenizer.sep_token
	if CLS is None or SEP is None:
		CLS, SEP = tokenizer.bos_token, tokenizer.eos_token
	if CLS is None:
		CLS = SEP
	return CLS, SEP

def to_device(x, gpuid):
	if gpuid == -1:
		return x.cpu()
	if x.device != gpuid:
		return x.cuda(gpuid)
	return x

def has_nan(t):
	return torch.isnan(t).sum() == 1

def tensor_on_dev(t, is_cuda):
	if is_cuda:
		return t.cuda()
	else:
		return t

def pick_label(dist):
	return np.argmax(dist, axis=1)

def torch2np(t, is_cuda):
	return t.numpy() if not is_cuda else t.cpu().numpy()

def save_opt(opt, path):
	with open(path, 'w') as f:
		f.write('{0}'.format(opt))


def last_index(ls, key):
	return len(ls) - 1 - ls[::-1].index(key)

def load_param_dict(path):
	# TODO, this is ugly
	f = h5py.File(path, 'r')
	return f


def save_param_dict(param_dict, path):
	file = h5py.File(path, 'w')
	for name, p in param_dict.items():
		file.create_dataset(name, data=p)

	file.close()


def load_dict(path):
	rs = {}
	with open(path, 'r+') as f:
		for l in f:
			if l.strip() == '':
				continue
			w, idx, cnt = l.strip().split()
			rs[int(idx)] = w
	return rs


def rand_tensor(shape, r1, r2):
	return (r1 - r2) * torch.rand(shape) + r2


def max_with_mask(v, dim):
	max_v, max_idx = v.max(dim)
	return max_v, max_idx, torch.zeros(v.shape).to(v).scatter(dim, max_idx.unsqueeze(dim), 1.0)

def min_with_mask(v, dim):
	min_v, min_idx = v.min(dim)
	return min_v, min_idx, torch.zeros(v.shape).to(v).scatter(dim, min_idx.unsqueeze(dim), 1.0)


def push_left(tensor, dim):
	mask = tensor == 0
	left_idx = mask.argsort(stable=True, dim=dim, descending=True)
	left = tensor.gather(dim=dim, index=left_idx)
	return left, left_idx


def shuffle(tensor, dim):
	rand_idx = torch.randperm(tensor.shape[dim], device=tensor.device)
	shuffled = tensor.index_select(dim=dim, index=rand_idx)
	assert(shuffled.shape == tensor.shape)
	idx_dims = list(1 for _ in tensor.shape)
	idx_dims[dim] = tensor.shape[dim]
	rand_idx = rand_idx.view(idx_dims).expand_as(tensor)
	return shuffled, rand_idx


def scatter(dim, index, src):
	x = torch.zeros(src.shape, dtype=src.dtype, device=src.device)
	x.scatter_(dim=dim, index=index, src=src)
	return x


def random_duplicate(mat):
 """ replace a nonzero integer by another nonzero randomly in each row. (credit Yuan Zhuang)"""
 #mat = torch.tensor([[3,0,0,2,0,3], [3,0,1,0,0,5]])

 # make sure each row as at least two valid elements.
 # for those without at least two, replace it with all-one temporarily
 valid_mask = (mat>0).sum(-1, keepdim=True) > 1
 mat_legit = torch.where(valid_mask, mat, 1)

 sampling_weight = (mat_legit > 0).float() # sampling probabolity. If an element is nonzero then it has sampling weight of 1 otherwise 0.

 sampled_indices = torch.multinomial(sampling_weight, 2) 
 
 src_idx =  sampled_indices[:,1] # indices to be duplicated 
 tgt_idx = sampled_indices[:,0] # indices to be replaced by the duplicated values
 
 # replace with duplicated values
 mat_legit[torch.arange(mat.size()[0]), tgt_idx] = mat_legit[torch.arange(mat.size()[0]), src_idx]
 result = torch.where(valid_mask, mat_legit, mat)
 return result


# use the idx (batch_l, seq_l, rs_l) (2nd dim) to select the middle dim of the content (batch_l, seq_l, d)
#	the result has shape (batch_l, seq_l, rs_l, d)
def batch_index2_select(content, idx, nul_idx):
	idx = idx.long()
	rs_l = idx.shape[-1]
	batch_l, seq_l, d = content.shape
	content = content.contiguous().view(-1, d)
	shift = torch.arange(0, batch_l).to(idx.device).long().view(batch_l, 1, 1)
	shift = shift * seq_l
	shifted = idx + shift
	rs = content[shifted].view(batch_l, seq_l, rs_l, d)
	#
	mask = (idx != nul_idx).unsqueeze(-1)
	return rs * mask.to(rs)

# use the idx (batch_l, rs_l) (1st dim) to select the middle dim of the content (batch_l, seq_l, d)
#	return (batch_l, rs_l, d)
def batch_index1_select(content, idx, nul_idx):
	idx = idx.long()
	rs_l = idx.shape[-1]
	batch_l, seq_l, d = content.shape
	content = content.contiguous().view(-1, d)
	shift = torch.arange(0, batch_l).to(idx.device).long().view(batch_l, 1)
	shift = shift * seq_l
	shifted = idx + shift
	rs = content[shifted].view(batch_l, rs_l, d)
	#
	mask = (idx != nul_idx).unsqueeze(-1)
	return rs * mask.to(rs)


def compose_log(orig_toks, role_labels, labels, transpose=True):
	role_labels = role_labels.numpy()
	seq_l = role_labels.shape[0]
	header = ['-' for _ in range(seq_l)]
	role_lines = []
	for i, row in enumerate(role_labels):
		roles = labels[row].tolist()
		roles = roles + ['O']	# TODO, the convert_role_labels prefers the last label to be O, so bit hacky here
		if is_v(roles[i]):	# on the diagonal, if the i-th label is a predicate, then tag it
			header[i] = orig_toks[i]
			roles, error_cnt = convert_role_labels(roles)
			role_lines.append(roles[:-1])
	log = [header] + role_lines
	log = np.asarray(log)
	# do a transpose
	if transpose:
		log = log.transpose((1, 0))
	
	rs = []
	for row in log:
		rs.append(' '.join(row))
	return '\n'.join(rs) + '\n'


def compose_semlink_log(semlink_r, semlink_a, srl_labels, vn_labels):
	log = ''
	for r, a in zip(semlink_r, semlink_a):
		log += f'{srl_labels[r]} - {vn_labels[a]}\n'
	return log


# convert role labels into conll format
#	for inconsistent BIO labels, it will hack to make sure of consistency
def convert_role_labels(labels):
	# firstly connect adjacent spans of the same label
	connected = [labels[0]]
	for i in range(1, len(labels)):
		prev, cur = labels[i-1], labels[i]
		if prev.startswith('I-') and cur.startswith('B-') and prev[2:] == cur[2:]:
			# force the current label to be I-*
			connected.append(prev)
		else:
			connected.append(cur)
	labels = connected

	inconsistent_cnt = 0
	rs = []
	cur_label = None
	for i, l in enumerate(labels):
		if l.startswith('B-'):
			if cur_label is not None:
				rs[-1] = '*)' if not rs[-1].startswith('(') else rs[-1] + ')'
			rs.append('({0}*'.format(l[2:]))
			cur_label = l[2:]
		elif l.startswith('I-'):
			if cur_label is not None:
				# if there is a inconsistency in label dependency, we fix it by treating this label as B- and end the previous label
				# 	this is a safeguard just in case, because the srl-eval.pl doesn't accept violation on that
				if cur_label != l[2:]:
					inconsistent_cnt += 1
					# take this label as B- then
					rs[-1] = '*)' if not rs[-1].startswith('(') else rs[-1] + ')'
					rs.append('({0}*'.format(l[2:]))
					#raise Exception('inconsistent labels: {0} {1}'.format(cur_l, l))
				else:
					rs.append('*' if i != len(labels)-1 else '*)')
			else:
				# take this label as B- then
				rs.append('({0}*'.format(l[2:]))
				#raise Exception('inconsistent labels: {0} {1}'.format(cur_l, l))
			cur_label = l[2:]
		elif l == 'O':
			if cur_label is not None:
				rs[-1] = '*)' if not rs[-1].startswith('(') else rs[-1] + ')'
			rs.append('*')
			cur_label = None
		else:
			raise Exception('unrecognized label {0}'.format(l))
	return rs, inconsistent_cnt


def system_call_eval(gold_path, pred_path):
	import subprocess
	rs = subprocess.check_output(['perl', 'srl-eval.pl', gold_path, pred_path])
	target_line = rs.decode('utf-8').split('\n')[6].split()
	f1 = float(target_line[-1]) * 0.01		# make percent to [0,1]
	return f1


if __name__ == '__main__':
	labels = 'O B-V I-V B-A0 B-A1 I-A1 O B-A3 I-A2 I-A2'.split()
	rs, _ = convert_role_labels(labels)
	print(labels)
	print(rs)