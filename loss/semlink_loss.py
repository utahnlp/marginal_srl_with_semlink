import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from util.holder import *
from util.util import *

# loss for semlink binding
class SemlinkLoss(torch.nn.Module):
	def __init__(self, name, opt, shared):
		super(SemlinkLoss, self).__init__()
		self.opt = opt
		self.shared = shared
		self.name = name
		
		self.one = torch.ones(1).float()
		if opt.gpuid != -1:
			self.one = to_device(self.one, self.opt.gpuid)

		self.violation_cnt = 0
		self.coverage_cnt = 0
		self.num_prop = 0
		self.num_ex = 0
		self.violation_log = []


	def forward(self, pack):
		v_label, v_l = self.shared.v_label, self.shared.v_l
		log_srl = pack['log_srl']
		log_vn = pack['log_vn']

		loss = torch.zeros(1)
		if self.opt.gpuid != -1:
			loss = to_device(loss, self.opt.gpuid)

		orig_l = self.shared.orig_seq_l
		max_orig_l = orig_l.max()
		semlink = self.shared.semlink
		semlink_l = self.shared.semlink_l
		num_prop = 0
		for i in range(self.shared.batch_l):
			v_i = v_label[i, :v_l[i]]
			semlink_l_i = semlink_l[i, :v_l[i]]	# (num_v,)
			max_semlink_l = semlink_l_i.max()
			if max_semlink_l <= 0:
				num_prop += v_l[i]
				continue

			semlink_mask_i = len_to_mask(semlink_l_i)	# (num_v, num_semlink)

			semlink_r_i = semlink[i, :v_l[i]][:, 0, :max_semlink_l]	# (num_v, num_semlink)
			semlink_a_i = semlink[i, :v_l[i]][:, 1, :max_semlink_l]	# (num_v, num_semlink)
			log_srl_i = log_srl[i, v_i, :orig_l[i]]	# (num_v, seq_l, num_srl_label)
			log_vn_i = log_vn[i, v_i, :orig_l[i]]	# (num_v, seq_l, num_vn_label)

			# (num_v, num_semlink, seq_l)
			log_semlink_r_i = batch_index1_select(content=log_srl_i.transpose(2, 1), idx=semlink_r_i, nul_idx=0)
			log_semlink_a_i = batch_index1_select(content=log_vn_i.transpose(2, 1), idx=semlink_a_i, nul_idx=0)

			# get loss at each position for its r and a probabilities
			loss_i = torch.abs(log_semlink_r_i - log_semlink_a_i)
			loss_i = loss_i * semlink_mask_i.unsqueeze(-1)

			loss = loss + loss_i.sum()
			num_prop += v_l[i]

		# stats
		# self.analyze(self.shared.srl_viterbi_pred, self.shared.vn_viterbi_pred)
		self.analyze_gold(self.shared.srl_label, self.shared.vn_label)

		# stats
		self.num_prop += int(num_prop)
		self.num_ex += self.shared.batch_l

		# # average over number of predicates or num_ex
		normalizer = float(num_prop) if self.opt.use_gold_predicate == 1 else sum([orig_l[i] for i in range(self.shared.batch_l)])
		#print(loss, normalizer)
		if normalizer == 0:
			assert(loss == 0)
			return loss, None
		#print('framrrole', loss / normalizer)
		return loss / normalizer, None


	def analyze(self, srl_pred_idx, vn_pred_idx):
		batch_l, _, _ = srl_pred_idx.shape
		orig_l = self.shared.orig_seq_l
		v_label = self.shared.v_label
		v_l = self.shared.v_l
		semlink_map = self.shared.semlink_map
		semlink_pool = self.shared.semlink_pool
		roleset_ids = self.shared.roleset_ids
		for i in range(self.shared.batch_l):
			v_i = v_label[i, :v_l[i]]
			srl_pred_i = srl_pred_idx[i, v_i, :orig_l[i]]	# (num_v, orig_l)
			vn_pred_i = vn_pred_idx[i, v_i, :orig_l[i]]		# (num_v, orig_l)
			roleset_i = roleset_ids[i, :v_l[i]]

			# do it predicate by predicate
			for k in range(v_l[i]):
				has_violation = False
				semlink_idx = semlink_map[vn_pred_i[k, v_i[k]], roleset_i[k]].item()
				if semlink_idx == -1:
					continue

				semlink_entry = semlink_pool[semlink_idx]	# (2, semlink_l)
				semlink_l = semlink_entry[0].count_nonzero().item()
				semlink_r = semlink_entry[0, :semlink_l]
				semlink_a = semlink_entry[1, :semlink_l]
				for r, a in zip(srl_pred_i[k], vn_pred_i[k]):
					# note here assumes no dup role/args exist in semlink for the same (vn_class, sense)
					if (r in semlink_r) != (a in semlink_a):
						has_violation = True
						break

				if has_violation:
					self.violation_cnt += 1


	def analyze_gold(self, srl_idx, vn_idx):
		batch_l, _, _ = srl_idx.shape
		orig_l = self.shared.orig_seq_l
		v_label = self.shared.v_label
		v_l = self.shared.v_l
		semlink = self.shared.semlink
		semlink_l = self.shared.semlink_l
		srl_labels = np.asarray(self.opt.srl_labels)
		vn_labels = np.asarray(self.opt.vn_labels)
		roleset_ids = self.shared.roleset_ids

		for i in range(self.shared.batch_l):
			v_i = v_label[i, :v_l[i]]
			semlink_l_i = semlink_l[i, :v_l[i]]	# (num_v,)
			semlink_r_i = semlink[i, :v_l[i]][:, 0, :]	# (num_v, num_semlink)
			semlink_a_i = semlink[i, :v_l[i]][:, 1, :]	# (num_v, num_semlink)
			srl_i = srl_idx[i, :v_l[i], :orig_l[i]]	# (num_v, orig_l)
			vn_i = vn_idx[i, :v_l[i], :orig_l[i]]		# (num_v, orig_l)
			roleset_ids_i = roleset_ids[i, :v_l[i]]

			# constructing log
			orig_tok_grouped = self.shared.res_map['orig_tok_grouped'][i][1:-1]
			orig_l_i = self.shared.orig_seq_l[i]
			a_gold_i = torch.zeros(orig_l_i, orig_l_i).long()	# O has idx 0
			r_gold_i = torch.zeros(orig_l_i, orig_l_i).long()
			for k, r_k in enumerate(srl_i):
				r_gold_i[v_i[k]] = r_k
			for k, a_k in enumerate(vn_i):
				a_gold_i[v_i[k]] = a_k

			
			srl_log = compose_log(orig_tok_grouped, r_gold_i[1:-1, 1:-1], srl_labels, transpose=False)
			vn_log = compose_log(orig_tok_grouped, a_gold_i[1:-1, 1:-1], vn_labels, transpose=False)
			for k in range(v_l[i]):
				if semlink_l_i[k] <= 0:
					continue
				has_violation = False
				self.coverage_cnt += 1

				r_pool = semlink_r_i[k, :semlink_l_i[k]]
				a_pool = semlink_a_i[k, :semlink_l_i[k]]
				vn_v = vn_i[k, v_i[k]]	# the vn class
				srl_v = srl_i[k, v_i[k]]
				semlink_log = compose_semlink_log(r_pool, a_pool, srl_labels, vn_labels)
				for r, a in zip(srl_i[k], vn_i[k]):
					# note here assumes no dup role/args exist in semlink for the same (vn_class, sense)
					if (r in r_pool) != (a in a_pool):
						has_violation = True
						break

				if has_violation:
					self.violation_cnt += 1
				#	self.violation_log.append((k, semlink_log, srl_log, vn_log))
				#	print('********'*10)
				#	print(f'Semlink violation at the predicate at {k} (starts from 0).')
				#	print('Tokens:')
				#	print(orig_tok_grouped)
				#	print(f'VN class {vn_labels[vn_v]}, SRL sense {self.opt.roleset[roleset_ids_i[k]]}')
				#	print('Semlink constraint:')
				#	print(semlink_log)
				#	print('SRL labels:')
				#	print(srl_log + '\n')
				#	print('VN labels:')
				#	print(vn_log + '\n')


	# return a string of stats
	def print_cur_stats(self):
		rho = (float(self.violation_cnt) / self.num_prop) if self.num_prop != 0 else 0.0
		coverage_ratio = (float(self.coverage_cnt) / self.num_prop) if self.num_prop != 0 else 0.0
		return "semlink rho: {0:.3f}, coverage ratio: {1:.3f}, num proposition {2}".format(rho, coverage_ratio, self.num_prop)
		#return ""

	# get training metric (scalar metric, extra metric)
	def get_epoch_metric(self):
		rho = (float(self.violation_cnt) / self.num_prop)  if self.num_prop != 0 else 0.0
		return rho, [rho]


	def begin_pass(self):
		self.violation_cnt = 0
		self.num_prop = 0
		self.num_ex = 0

	def end_pass(self):
		pass
