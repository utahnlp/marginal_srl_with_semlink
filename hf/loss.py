import sys
import torch
from torch import nn
import numpy as np
from .crf import *
from .util import *
from collections import defaultdict

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

# Cross CRF loss that does cross product of VN and SRL labels and condense them into one CRF
class CrossCRFLoss(torch.nn.Module):
	def __init__(self, opt, shared):
		super(CrossCRFLoss, self).__init__()
		self.opt = opt
		self.shared = shared
		self.name = opt.loss

		self.vn_labels = np.asarray(self.opt.vn_labels)
		self.srl_labels = np.asarray(self.opt.srl_labels)
		self.condensed_labels = np.asarray(self.opt.condensed_labels)
		self.cross2condensed = torch.tensor(self.opt.cross2condensed, dtype=torch.long)
		self.condensed2cross = torch.tensor(self.opt.condensed2cross, dtype=torch.long)
		self.condensed2srl = torch.tensor(self.opt.condensed2srl, dtype=torch.long)

		self.vn_classes = np.asarray(self.opt.vn_classes)
		self.roleset_suffix = np.asarray(self.opt.roleset_suffix)
		self.roleset = np.asarray(self.opt.roleset)

		# only mark the B-* labels
		self.condensed_b_mask = [1 if l.startswith('B-') else 0 for l in self.opt.condensed_labels]
		self.condensed_b_mask = torch.Tensor(self.condensed_b_mask)

		# only marks B-* and non-O labels
		self.condensed_content_mask = [0 for _ in self.opt.condensed_labels]
		# only marks B-* for core SRL labels
		self.condensed_core_masks = [[0 for _ in self.opt.condensed_labels] for _ in range(6)]	# ARG0-ARG5
		for i in range(1, len(self.opt.vn_labels)):
			for k in range(1, len(self.opt.srl_labels)):
				p = self.cross2condensed[i][k]
				if p == -1 or self.opt.vn_labels[i].split('-')[-1] == 'V' or self.opt.srl_labels[k].split('-')[-1] == 'V':
					continue
				#if self.condensed_labels[p].startswith('B-'):
				#	self.condensed_b_mask[p] = 1
				if self.opt.vn_labels[i] != 'O' and self.opt.srl_labels[k] != 'O':
					self.condensed_content_mask[p] = 1
				for c in range(6):
					if f'ARG{c}' in self.condensed_labels[p]:
						self.condensed_core_masks[c][p] = 1
		self.condensed_content_mask = torch.tensor(self.condensed_content_mask)
		self.condensed_core_masks = torch.tensor(self.condensed_core_masks)

		self.srl_core_masks = [[0 for _ in self.opt.srl_labels] for _ in range(6)] 	# ARG0-ARG5
		self.srl_core_label_idx = []
		for i, srl_l in enumerate(self.opt.srl_labels):
			if not srl_l.startswith('B-'):
				continue
			for c in range(6):
				if f'ARG{c}' in srl_l:
					self.srl_core_masks[c][i] = 1
					self.srl_core_label_idx.append(i)
		self.srl_core_masks = torch.tensor(self.srl_core_masks)
		print('srl_core_labels', [self.opt.srl_labels[k] for k in self.srl_core_label_idx])

		# (num_vn_label, num_condensed_label)
		self.vn2condensed_mask = [[0 for _ in self.opt.condensed_labels] for _ in self.opt.vn_labels]
		for i in range(len(self.opt.vn_labels)):
			for k in range(len(self.opt.srl_labels)):
				p = self.cross2condensed[i][k]
				if p != -1:
					self.vn2condensed_mask[i][p] = 1

			if sum(self.vn2condensed_mask[i]) == 0:
				print(f'{self.opt.vn_labels[i]} has no mapping to condensed labels')

		self.vn2condensed_mask = torch.tensor(self.vn2condensed_mask, dtype=torch.bool)
		self.vn2condensed_mask = self.vn2condensed_mask
		num_valid_vn = self.vn2condensed_mask.any(-1).sum().item()
		print('num_valid_vn', f'{num_valid_vn}/{len(self.vn_labels)}')

		# (num_srl_label, num_condensed_label)
		self.srl2condensed_mask = [[0 for _ in self.opt.condensed_labels] for _ in self.opt.srl_labels]
		for k in range(len(self.opt.srl_labels)):
			for i in range(len(self.opt.vn_labels)):
				p = self.cross2condensed[i][k]
				if p != -1:
					self.srl2condensed_mask[k][p] = 1

			if sum(self.srl2condensed_mask[k]) == 0:
				print(f'{self.opt.srl_labels[k]} has no mapping to condensed labels')

		self.srl2condensed_mask = torch.tensor(self.srl2condensed_mask, dtype=torch.bool)
		self.srl2condensed_mask = self.srl2condensed_mask
		num_valid_srl = self.srl2condensed_mask.any(-1).sum().item()
		print('num_valid_srl', f'{num_valid_srl}/{len(self.srl_labels)}')

		self.vn_b2i = [0 for _ in self.opt.vn_labels]
		self.srl_b2i = [0 for _ in self.opt.srl_labels]
		for i, label in enumerate(self.opt.vn_labels):
			if label.startswith('B-'):
				label = label[2:]
				if ('I-' + label) in self.opt.vn_labels:
					self.vn_b2i[i] = self.opt.vn_labels.index('I-' + label)
		for i, label in enumerate(self.opt.srl_labels):
			if label.startswith('B-'):
				label = label[2:]
				if ('I-' + label) in self.opt.srl_labels:
					self.srl_b2i[i] = self.opt.srl_labels.index('I-' + label)
		self.vn_b2i = torch.tensor(self.vn_b2i)
		self.srl_b2i = torch.tensor(self.srl_b2i)


		# make CRF module
		constraints = allowed_transitions("BIO", self.opt.condensed_map_inv)
		self.crf = ConditionalRandomField(num_tags=len(self.condensed_labels), constraints=constraints)


	def map_cross_to_condensed(self, vn_seq, srl_seq):
		role_label = self.cross2condensed[vn_seq, srl_seq]
		# replace missed entries (-1) with O
		return torch.where(role_label != -1, role_label, 0)


	def map_condensed_to_cross(self, seq):
		pairs = self.condensed2cross[seq]
		return pairs[..., 0], pairs[..., 1]


	def get_semlink_penalty_mask(self, a_semlink):
		num_v = a_semlink.shape[0]
		condensed_l = len(self.opt.condensed_labels)
		assert(len(a_semlink.shape) == 3)	# (num_v, 2, max_num_semlink)
		# importantly, replace -1 with index that points to O (ie 0)
		b_roles = a_semlink[:, 0, :]
		b_roles = torch.where(b_roles != -1, b_roles, 0)	# (num_v, max_num_semlink)
		i_roles = self.srl_b2i[b_roles].to(b_roles.device)
		b_args = a_semlink[:, 1, :]
		b_args = torch.where(b_args != -1, b_args, 0)		# (num_v, max_num_semlink)
		i_args = self.vn_b2i[b_args].to(b_args.device)
		# (num_v, max_num_semlink*2)
		roles = torch.cat([b_roles, i_roles], dim=-1)
		args = torch.cat([b_args, i_args], dim=-1)
		# (num_v, max_num_semlink*2, condensed_l)
		srl_mask = self.srl2condensed_mask[roles].to(a_semlink.device)
		vn_mask = self.vn2condensed_mask[args].to(a_semlink.device)

		# Inner mask denotes the explicit content label pairs mentioned in semlink.
		inner_mask = torch.logical_and(srl_mask, vn_mask)
		# each of the semlink corresponds to exactly one label in the condensed list.
		assert(torch.all(inner_mask.int().sum(-1) <= 1))
		# (num_v, condensed_l)
		inner_mask = inner_mask.logical_and(self.condensed_content_mask.view(1, 1, condensed_l).to(a_semlink.device))
		inner_mask = inner_mask.any(1)

		# What we want is that content label pairs not in the inner mask are disabled, e.g.,
		# 	if semlink defines ARG0-AGENT and ARG1-THEME, then ARG0-non-AGENT, ARG1-non-THEME, non-ARG0-AGENT, and non-ARG1-THEME are blocked.
		#	ARG2 is only allowed to O
		disable_mask = inner_mask.logical_not().logical_and(self.condensed_content_mask.view(1, condensed_l).to(a_semlink.device))
		# Furthermore, if the incoming semlink is empty, we disable nothing
		valid_semlink = (roles == 0).all(-1).logical_not().view(num_v, 1)
		disable_mask = disable_mask.logical_and(valid_semlink)

		# (num_v, condensed_l)
		return disable_mask


	def is_active(self):
		return 'cross' in self.shared.batch_flag


	def forward(self, pack):
		batch_l = self.shared.batch_l
		v_label, v_l = self.shared.v_label, self.shared.v_l
		flat_score = pack['label_score']
		flat_log_score = pack['log_score']

		num_v = flat_score.shape[0]
		orig_l = self.shared.orig_seq_l
		max_orig_l = orig_l.max()
		semlink_l = self.shared.semlink_l
		semlink = self.shared.semlink[:, :v_l.max(), :, :semlink_l.max()]

		flat_semlink = []
		flat_mask = []
		# pack everything into (num_v, max_orig_l, ...)
		for i in range(batch_l):
			mask_i = torch.zeros(v_l[i], max_orig_l).byte()
			mask_i[:, :orig_l[i]] = True

			flat_mask.append(mask_i)
			flat_semlink.append(semlink[i, :v_l[i]])

		flat_score = flat_score[:, :max_orig_l]
		flat_mask = torch.cat(flat_mask, dim=0).to(flat_score.device)
		flat_semlink = torch.cat(flat_semlink, dim=0).to(flat_score.device)
		flat_pred = flat_log_score[:, :max_orig_l].argmax(-1)

		semlink_penalty_mask = self.get_semlink_penalty_mask(flat_semlink)
		flat_score += semlink_penalty_mask.unsqueeze(1) * self.opt.neg_inf

		decoded = self.crf.viterbi_tags(flat_score, flat_mask)

		# unpack decoded to two (batch_l, num_v, max_orig_l) tensors
		# one for vn, one for srl.
		flat_vn_pred = torch.zeros(num_v, max_orig_l).long()
		flat_srl_pred = torch.zeros(num_v, max_orig_l).long()
		flat_vn_class = torch.zeros(num_v).long()
		flat_roleset_suffix = torch.zeros(num_v).long()
		flat_roleset_ids = torch.zeros(num_v).long()
		start = 0
		for i in range(batch_l):
			for k in range(v_l[i]):
				p = torch.Tensor(decoded[start][0]).long()
				vn_seq, srl_seq = self.map_condensed_to_cross(p)
	
				flat_vn_pred[start, :orig_l[i]] = vn_seq
				flat_srl_pred[start, :orig_l[i]] = srl_seq
				flat_vn_class[start] = self.shared.vn_class[i, k]
				flat_roleset_suffix[start] = self.shared.roleset_suffixes[i, k]
				flat_roleset_ids[start] = self.shared.roleset_ids[i, k]
				start += 1
		assert(start == len(decoded))
		flat_vn_pred = flat_vn_pred.to(flat_score.device)
		flat_srl_pred = flat_srl_pred.to(flat_score.device)

		pretty_vn_pred, _ = self.record_log('vn', self.vn_labels, flat_vn_pred, bv_idx=self.opt.vn_labels.index('B-V'))
		pretty_srl_pred, _ = self.record_log('srl', self.srl_labels, flat_srl_pred, bv_idx=self.opt.srl_labels.index('B-V'))

		return {
			# 'flat_vn_pred': flat_vn_pred, 'flat_srl_pred': flat_srl_pred,
			'vn_output': pretty_vn_pred, 'srl_output': pretty_srl_pred
			}


	def record_log(self, log_name, label_names, flat_pred_idx, bv_idx):
		batch_l = self.shared.batch_l
		orig_l = self.shared.orig_seq_l
		v_label = self.shared.v_label
		v_l = self.shared.v_l

		pretty_role_log, pretty_orig_toks = [], []

		start = 0
		for i in range(batch_l):
			# do analysis without cls and sep
			orig_toks = self.shared.orig_toks[i]
			orig_l_i = orig_l[i].item()	# convert to scalar
			v_i = v_label[i, :v_l[i]]
			end = start + v_l[i]

			a_pred_i = flat_pred_idx[start:end, :orig_l_i].cpu()
			# only keep predictions on the predicate positions
			a_pred_i_new = torch.zeros(orig_l_i, orig_l_i).long()	# O has idx 0
			for k in range(v_l[i]):
				a_pred_i_new[v_i[k]] = a_pred_i[k]
				# force it to be the vn class
				# TODO, double check the coverage is correct
				if self.opt.use_gold_predicate == 1:
					a_pred_i_new[v_i[k], v_i[k]] = bv_idx
			a_pred_i = a_pred_i_new

			pretty_role_log.append(self.compose_log(label_names, orig_toks[1:-1], a_pred_i[1:-1, 1:-1]))
			pretty_orig_toks.append(orig_toks[1:-1])
			start += v_l[i]
		return pretty_role_log, pretty_orig_toks

	# compose log for one example
	#	role_labels of shape (seq_l, seq_l)
	def compose_log(self, label_names, orig_toks, role_labels):
		role_labels = role_labels.numpy()
		seq_l = role_labels.shape[0]

		header = ['-' for _ in range(seq_l)]
		role_lines = []
		for i, row in enumerate(role_labels):
			roles = label_names[row].tolist()
			roles = roles + ['O']	# TODO, the convert_role_labels prefers the last label to be O, so bit hacky here
			if is_v(roles[i]):	# on the diagonal, if the i-th label is a predicate, then tag it
				header[i] = orig_toks[i]
				roles, bio_error_cnt = convert_role_labels(roles)
				role_lines.append(roles[:-1])	# skip the padded O

		role_lines = [' '.join(p) for p in role_lines]
		return role_lines



if __name__ == '__main__':
	pass




