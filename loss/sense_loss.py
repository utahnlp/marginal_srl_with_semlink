import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from util.holder import *
from util.util import *

# loss for roleset suffix (sense)
class SenseLoss(torch.nn.Module):
	def __init__(self, name, opt, shared):
		super(SenseLoss, self).__init__()
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
		return 'srl' in self.shared.batch_flag

	def forward(self, pack):
		# v_label stores the indices of verbs.
		v_label, v_l = self.shared.v_label, self.shared.v_l
		all_sense_score = pack['sense_score']	# (batch_l, max_orig_l, num_label)
		roleset_suffixes = self.shared.roleset_suffixes 	# (batch_l, max_v_num)
		zero = to_device(torch.tensor(0.0, dtype=torch.float32), self.opt.gpuid)	# somehow needs float32
		num_suffix = len(self.opt.roleset_suffix)

		max_v_l = v_l.max()
		# (batch_l, max_v_l)
		sense_gold = roleset_suffixes[:, :max_v_l].contiguous()
		# (batch_l, max_v_l, num_suffix)
		sense_score = torch.take_along_dim(all_sense_score, v_label[:, :max_v_l].view(-1, max_v_l, 1), dim=1)
		# (batch_l, max_v_l)
		valid_sense_mask = to_device(len_to_mask(v_l), self.opt.gpuid)
		# (batch_l, max_v_l, num_suffix)
		sense_score_mask = valid_sense_mask.view(-1, max_v_l, 1).expand(-1, max_v_l, num_suffix)
		# mask invalid scores to be -inf
		sense_score = sense_score * sense_score_mask + sense_score_mask.logical_not() * self.opt.neg_inf
		log_sense_score = nn.LogSoftmax(-1)(sense_score)

		# (batch_l, max_v_l)
		loss = torch.nn.NLLLoss(reduction='none')(
			log_sense_score.view(-1, num_suffix),
			sense_gold.view(-1)).view(self.shared.batch_l, max_v_l)

		# mask out loss for those invalid verbs.
		# need to use torch.where instead of *, otherwise nan loss
		loss = torch.where(valid_sense_mask, loss, zero)
		loss = loss.sum()

		# stats
		num_prop = valid_sense_mask.sum().item()
		self.num_prop += num_prop
		self.num_ex += self.shared.batch_l

		self.analyze(sense_score.argmax(-1), sense_gold, valid_sense_mask)

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
		return 'roleset suffix (sense) accuracy: {0:.3f}'.format(self.num_correct / normalizer)


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
