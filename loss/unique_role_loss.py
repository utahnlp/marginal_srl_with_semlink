import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from util.holder import *
from util.util import *

# loss on unique core argument
class UniqueRoleLoss(torch.nn.Module):
	def __init__(self, name, opt, shared):
		super(UniqueRoleLoss, self).__init__()
		self.opt = opt
		self.shared = shared
		self.name = name
		
		self.one = torch.ones(1).float()
		if opt.gpuid != -1:
			self.one = to_device(self.one, self.opt.gpuid)

		self.violation_cnt = 0
		self.num_prop = 0
		self.num_ex = 0

		self.core_labels = []
		self.core_label_idx = []
		for idx, l in self.opt.srl_label_map_inv.items():
			if l.startswith('B-ARG') and l[5].isnumeric():	# only applies B-* tags for core args
				self.core_labels.append(l)
				self.core_label_idx.append(idx)
		print('unique role constraint applies to: ', self.core_labels)


	def is_active(self):
		return 'cross' in self.shared.batch_flag


	def forward(self, pack):
		batch_l, v_l = self.shared.batch_l, self.shared.v_l
		log_pa = pack['log_srl']

		orig_l = self.shared.orig_seq_l
		max_orig_l = orig_l.max()
		num_prop, source_l, num_srl_label = log_pa.shape

		lengths = []
		for i in range(batch_l):
			lengths.extend([orig_l[i]] * v_l[i])
		assert(len(lengths) == num_prop)
		lengths = torch.Tensor(lengths).long().to(log_pa)
		# len_mask marks the valid (non-padding) tokens
		len_mask = len_to_mask(lengths, max_orig_l)	# num_v, orig_l
		len_mask_ext = len_mask.unsqueeze(-1).logical_and(len_mask.unsqueeze(1))	# num_v, orig_l, orig_l
		# diag_mask handles the diag up to each sequence length
		diag_mask = torch.eye(max_orig_l).to(log_pa)
		diag_mask = diag_mask.unsqueeze(0).logical_and(len_mask_ext)	# num_v, orig_l, orig_l
		# mark the diag and the padding parts to be a large positive number
		mask = diag_mask.logical_or(len_mask_ext.logical_not()).float()
		mask *= 10000
		
		log_core = log_pa[:, :max_orig_l, self.core_label_idx]	# num_v, orig_l, num_core_label
		log_core_ext = log_core.unsqueeze(1).expand(-1, max_orig_l, -1, -1)	# num_v, orig_l, orig_l, num_core_label

		log_neg_core_ext = (self.one - log_core_ext.exp()).clamp(min=1e-6).log()	# num_v, orig_l, orig_l, num_core_label
		log_neg_core_ext = log_neg_core_ext + mask.unsqueeze(-1)

		loss = torch.relu(log_core - log_neg_core_ext.min(2)[0])
		loss = loss * len_mask.unsqueeze(-1)
		loss = loss.sum()

		self.analyze(self.shared.flat_srl_pred)

		# stats
		self.num_prop += int(num_prop)

		return loss / num_prop, None


	def analyze(self, pred_idx):
		num_v, orig_l = pred_idx.shape
		for i in range(num_v):
			pred_idx_i = pred_idx[i]
			acc = None
			for k, label in enumerate(self.core_label_idx):
				dup_core_cnt = ((pred_idx_i == label).sum(-1) > 1).int()	# (num_v)
				acc = dup_core_cnt if k == 0 else (acc + dup_core_cnt)
			self.violation_cnt += (acc > 0).sum().item()


	# return a string of stats
	def print_cur_stats(self):
		rho = float(self.violation_cnt) / self.num_prop
		return "unique rho: {0:.3f}".format(rho)
		#return ""

	# get training metric (scalar metric, extra metric)
	def get_epoch_metric(self):
		rho = float(self.violation_cnt) / self.num_prop
		return rho, [rho]
		#return None, []


	def begin_pass(self):
		self.violation_cnt = 0
		self.num_prop = 0

	def end_pass(self):
		pass
