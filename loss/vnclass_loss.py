import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from util.holder import *
from util.util import *

# loss for verbnet class
class VNClassLoss(torch.nn.Module):
	def __init__(self, name, opt, shared):
		super(VNClassLoss, self).__init__()
		self.opt = opt
		self.shared = shared
		self.name = name
		
		self.one = torch.ones(1).float()
		if opt.gpuid != -1:
			self.one = to_device(self.one, self.opt.gpuid)

		self.num_prop = 0
		self.num_ex = 0
		self.num_correct = 0


	def is_active(self):
		return 'vn' in self.shared.batch_flag


	def forward(self, pack):
		# v_label stores the indices of verbs.
		v_label, v_l = self.shared.v_label, self.shared.v_l
		all_v_score = pack['vnclass_score']	# (batch_l, max_orig_l, num_vn_class)
		label_mask = to_device(torch.tensor(self.opt.v_mask), self.opt.gpuid)
		zero = to_device(torch.tensor(0.0, dtype=torch.float32), self.opt.gpuid)	# somehow needs float32

		max_v_l = v_l.max()
		# (batch_l, max_v_l, 1)
		#v_gold = torch.take_along_dim(self.shared.vn_label[:, :max_v_l], v_label[:, :max_v_l].view(-1, max_v_l, 1), dim=2)
		# (batch_l, max_v_l)
		#v_gold = v_gold.squeeze(-1)
		v_gold = to_device(self.shared.vn_class, self.opt.gpuid)
		# (batch_l, max_v_l, num_vn_class)
		v_score = torch.take_along_dim(all_v_score, v_label[:, :max_v_l].view(-1, max_v_l, 1), dim=1)
		# (batch_l, max_v_l)
		valid_v_mask = to_device(len_to_mask(v_l), self.opt.gpuid)
		# (batch_l, max_v_l, num_vn_class)
		v_score_mask = valid_v_mask.view(-1, max_v_l, 1).expand(-1, -1, self.opt.num_vn_class)
		# mask invalid scores to be -inf
		v_score = v_score * v_score_mask + v_score_mask.logical_not() * self.opt.neg_inf
		log_v_score = nn.LogSoftmax(-1)(v_score)

		# (batch_l, max_v_l)
		loss = torch.nn.NLLLoss(reduction='none')(
			log_v_score.view(-1, self.opt.num_vn_class),
			v_gold.view(-1)).view(self.shared.batch_l, max_v_l)

		# mask out loss for those invalid verbs.
		# need to use torch.where instead of *, otherwise nan loss
		loss = torch.where(valid_v_mask, loss, zero)
		loss = loss.sum()

		# stats
		num_prop = valid_v_mask.sum().item()
		self.num_prop += num_prop
		self.num_ex += self.shared.batch_l

		self.analyze(v_score.argmax(-1), v_gold, valid_v_mask)

		normalizer = max(num_prop, 1.0)
		# # average over number of predicates or num_ex
		return loss / normalizer, None


	def analyze(self, pred_idx, gold_idx, mask):
		num_correct = (pred_idx == gold_idx)
		num_correct *= mask
		self.num_correct += num_correct.sum().item()


	# return a string of stats
	def print_cur_stats(self):
		normalizer = max(self.num_prop, 1)
		return 'vnclass accuracy: {0:.3f}'.format(self.num_correct / normalizer)


	# get training metric (scalar metric, extra metric)
	def get_epoch_metric(self):
		normalizer = max(self.num_prop, 1)
		f1 = (self.num_correct / normalizer)
		return f1, [f1]


	def begin_pass(self):
		self.num_prop = 0
		self.num_ex = 0
		self.num_correct = 0


	def end_pass(self):
		pass
