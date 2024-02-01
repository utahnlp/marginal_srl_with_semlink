import sys
import torch
from torch import nn
from torch.autograd import Variable
from util.holder import *
from util.util import *
from .crf import *
import numpy as np

# CRF loss function incl. decoding
class CRFLoss(torch.nn.Module):
	def __init__(self, name, opt, shared):
		super(CRFLoss, self).__init__()
		self.opt = opt
		self.shared = shared
		self.name = name

		self.vn_labels = np.asarray(self.opt.vn_labels)
		self.srl_labels = np.asarray(self.opt.srl_labels)

		self.vn_classes = np.asarray(self.opt.vn_classes)
		self.roleset_suffix = np.asarray(self.opt.roleset_suffix)
		self.roleset = np.asarray(self.opt.roleset)

		# (num_vn_label, num_condensed_label)
		self.vn2condensed_mask = [[0 for _ in self.opt.condensed_labels] for _ in self.opt.vn_labels]
		for i in range(len(self.opt.vn_labels)):
			for k in range(len(self.opt.srl_labels)):
				p = self.opt.cross2condensed[i][k]
				if p != -1:
					self.vn2condensed_mask[i][p] = 1

			if sum(self.vn2condensed_mask[i]) == 0:
				print(f'{self.opt.vn_labels[i]} has no mapping to condensed labels')

		self.vn2condensed_mask = torch.tensor(self.vn2condensed_mask, dtype=torch.bool)
		self.vn2condensed_mask = to_device(self.vn2condensed_mask, self.opt.gpuid)
		num_valid_vn = self.vn2condensed_mask.any(-1).sum().item()
		print('num_valid_vn', f'{num_valid_vn}/{len(self.vn_labels)}')

		# (num_srl_label, num_condensed_label)
		self.srl2condensed_mask = [[0 for _ in self.opt.condensed_labels] for _ in self.opt.srl_labels]
		for k in range(len(self.opt.srl_labels)):
			for i in range(len(self.opt.vn_labels)):
				p = self.opt.cross2condensed[i][k]
				if p != -1:
					self.srl2condensed_mask[k][p] = 1

			if sum(self.srl2condensed_mask[k]) == 0:
				print(f'{self.opt.srl_labels[k]} has no mapping to condensed labels')

		self.srl2condensed_mask = torch.tensor(self.srl2condensed_mask, dtype=torch.bool)
		self.srl2condensed_mask = to_device(self.srl2condensed_mask, self.opt.gpuid)
		num_valid_srl = self.srl2condensed_mask.any(-1).sum().item()
		print('num_valid_srl', f'{num_valid_srl}/{len(self.srl_labels)}')


		if 'vn' in self.name:
			self.label_map_inv = self.opt.vn_label_map_inv
			self.labels = self.opt.vn_labels
			self.num_label = self.opt.num_vn_label
			self.log_name = 'vn'
			self.class_names = self.vn_classes
		elif 'srl' in self.name:
			self.label_map_inv = self.opt.srl_label_map_inv
			self.labels = self.opt.srl_labels
			self.num_label = self.opt.num_srl_label
			self.log_name = 'srl'
			# self.class_names = self.roleset_suffix
			self.class_names = self.roleset
		else:
			raise Exception('need to specify either vn or srl in crf loss name', self.name)

		self.label_groups = []
		self.label_group_map = {}
		for idx, l in self.label_map_inv.items():
			group = l[2:] if l != 'O' else l
			if group not in self.label_groups:
				self.label_groups.append(group)
			self.label_group_map[idx] = self.label_groups.index(group)

		self.labels = np.asarray(self.labels)

		constraints = allowed_transitions("BIO", self.label_map_inv)

		self.crf = ConditionalRandomField(num_tags=self.num_label, constraints=constraints, gpuid=opt.gpuid)

		self.quick_acc_sum = 0.0
		self.num_ex = 0
		self.num_prop = 0
		self.inconsistent_bio_cnt = 0

		self.log_types = []
		if hasattr(self.opt, 'logs'):
			self.log_types = self.opt.logs.strip().split(',')

		self.filtered_sent = set()
		self.sent_filter = set()
		if hasattr(self.opt, 'sent_filter') and self.opt.sent_filter != self.opt.dir:
			with open(self.opt.sent_filter, 'r') as f:
				for l in f:
					if l.strip() == '':
						continue
					self.sent_filter.add(l.strip().lower())

	# the decode function is for the demo where gold predicate might not present
	def decode(self, pack, v_label=None, v_l=None):
		if 'vn' in self.name:
			log_pa = pack['log_vn']
			score = pack['vn_score']
		elif 'srl' in self.name:
			log_pa = pack['log_srl']
			score = pack['srl_score']
		else:
			raise Exception('need to specify either vn or srl in crf loss name', self.name)

		batch_l, source_l, _, _ = score.shape
		orig_l = self.shared.orig_seq_l
		max_orig_l = orig_l.max()

		a_score = []
		a_mask = []

		if v_label is None:
			v_label_pred = to_device(torch.zeros(batch_l, max_orig_l).long(), self.opt.gpuid)
			v_l_pred = to_device(torch.zeros(batch_l).long(), self.opt.gpuid)

			# use v classifier to get predicates
			for i in range(batch_l):
				log_v_i = pack['log_v'][i, :orig_l[i]]
				v_pred_i = log_v_i.argmax(-1)
				v_label_i = torch.tensor([k if self.v_mask[k] else 0 for k in v_pred_i]).nonzero(as_tuple=False).view(-1)
				v_l_pred[i] = len(v_label_i)
				v_label_pred[i, :v_l_pred[i]] = v_label_i

				# use heurisitics to get a predicate
				if v_l_pred[i] == 0:
					v_mask = self.v_mask.view(1, -1)
					log_v_i = log_v_i * v_mask + (1 - v_mask) * -1e6
					v_label_pred[i, 0] = log_v_i.max(-1)[0].argmax(-1).view(1)
					v_l_pred[i] = 1

			v_label = v_label_pred
			v_l = v_l_pred

		else:
			v_label = to_device(v_label, self.opt.gpuid)
			v_l = to_device(v_l, self.opt.gpuid)

		# pack everything into (batch_l*acc_orig_l, max_orig_l, ...)
		for i in range(batch_l):
			a_mask_i = torch.zeros(v_l[i], max_orig_l).byte()
			a_mask_i[:, :orig_l[i]] = True
			a_mask.append(a_mask_i)
			
			a_score_i = score[i].index_select(0, v_label[i, :v_l[i]])[:, :max_orig_l]
			a_score.append(a_score_i)

		a_score = to_device(torch.cat(a_score, dim=0), self.opt.gpuid)
		a_mask = to_device(torch.cat(a_mask, dim=0), self.opt.gpuid)

		decoded = self.crf.viterbi_tags(a_score, a_mask)

		# unpack pred_idx to (batch_l, max_orig_l, max_orig_l, ...)
		pred_idx = torch.zeros(batch_l, max_orig_l, max_orig_l).long()
		row_idx = 0
		for i in range(batch_l):
			for k in range(v_l[i]):
				pred_idx[i, v_label[i, k], :orig_l[i]] = torch.Tensor(decoded[row_idx][0]).long()
				row_idx += 1
		assert(row_idx == len(decoded))
		pred_idx = to_device(pred_idx, self.opt.gpuid)

		return pred_idx, {'v_label': v_label, 'v_l': v_l}


	def is_active(self):
		return 'vn' in self.shared.batch_flag or 'srl' in self.shared.batch_flag


	def forward(self, pack):
		batch_l = self.shared.batch_l
		orig_l = self.shared.orig_seq_l
		max_orig_l = orig_l.max()
		v_label, v_l = self.shared.v_label, self.shared.v_l

		if self.name == 'vn_crf':
			flat_log = pack['log_vn']
			flat_score = pack['vn_score']
			role_label = self.shared.vn_label
			class_idx = self.shared.vn_class
			# if this batch has no vn labels, then skip
			if role_label is None:
				return torch.zeros(1, device=log_pa.device), None

		elif self.name == 'srl_crf':
			flat_log = pack['log_srl']
			flat_score = pack['srl_score']
			role_label = self.shared.srl_label
			# class_idx = self.shared.roleset_suffixes
			class_idx = self.shared.roleset_ids
			# if this batch has no srl labels, then skip
			if role_label is None:
				return torch.zeros(1, device=log_pa.device), None
		else:
			raise Exception('need to specify either vn or srl in crf loss name', self.name)

		num_v = flat_score.shape[0]

		flat_gold_srl = []
		flat_gold = []
		flat_mask = []
		# pack everything into (num_v, max_orig_l, ...)
		for i in range(batch_l):
			gold_i = torch.zeros(v_l[i], max_orig_l).long()	# O has idx 0
			gold_i[:, :orig_l[i]] = role_label[i, :v_l[i], :orig_l[i]]	# (num_v, orig_l)
			gold_srl_i = torch.zeros(v_l[i], max_orig_l).long()
			gold_srl_i[:, :orig_l[i]] = self.shared.srl_label[i, :v_l[i], :orig_l[i]]
			mask_i = torch.zeros(v_l[i], max_orig_l).byte()
			mask_i[:, :orig_l[i]] = True
			flat_mask.append(mask_i)
			flat_gold.append(gold_i)
			flat_gold_srl.append(gold_srl_i)

		flat_score = flat_score[:, :max_orig_l]
		flat_mask = to_device(torch.cat(flat_mask, dim=0), self.opt.gpuid)
		flat_gold = to_device(torch.cat(flat_gold, dim=0), self.opt.gpuid)
		flat_gold_srl = to_device(torch.cat(flat_gold_srl, dim=0), self.opt.gpuid)
		flat_pred = flat_log[:, :max_orig_l].argmax(-1)

		if self.shared.is_train:
			loss = self.crf(flat_score, flat_gold, flat_mask)
		else:
			# in validation mode, no need to count the loss here
			loss = to_device(torch.zeros(1), self.opt.gpuid)

			if self.opt.eval_with_srl == 1 and self.name == 'vn_crf':
				flat_srl_beam = self.srl2condensed_mask[flat_gold_srl]	# (num_v, seq_l, condensed_l)
				flat_srl_beam = flat_srl_beam.unsqueeze(2)
				flat_vn2condensed = self.vn2condensed_mask.view(1, 1, len(self.opt.vn_labels), len(self.opt.condensed_labels))
				flat_vn_beam = flat_srl_beam.logical_and(flat_vn2condensed).any(dim=-1)	# (num_v, seq_l, num_vn)
				flat_score += flat_vn_beam.logical_not() * self.opt.neg_inf

			decoded = self.crf.viterbi_tags(flat_score, flat_mask)
			flat_pred = torch.zeros(num_v, max_orig_l).long()
			flat_v_label = torch.zeros(num_v).long()
			start = 0
			for i in range(batch_l):
				for k in range(v_l[i]):
					flat_pred[start, :orig_l[i]] = torch.Tensor(decoded[start][0]).long()
					flat_v_label[start] = class_idx[i, k]
					start += 1
			assert(start == len(decoded))
			flat_pred = to_device(flat_pred, self.opt.gpuid)

		# some quick analyses
		batch_acc_sum = self._quick_acc(flat_pred, flat_gold) * num_v
		self.quick_acc_sum += batch_acc_sum
		self.num_ex += batch_l
		self.num_prop += num_v

		if not self.shared.is_train:
			self.record_log(self.log_name, self.labels, self.class_names, flat_v_label, flat_pred, flat_gold)

		return loss / num_v, {'flat_pred': flat_pred, 'flat_gold': flat_gold}


	# return rough estimate of accuracy that counts only non-O elements (in both pred and gold)
	def _quick_acc(self, pred_idx, gold_idx):
		pred_mask = pred_idx != 0
		gold_mask = gold_idx != 0
		non_o_mask = ((pred_mask + gold_mask) > 0).int()
		overlap = (pred_idx == gold_idx).int() * non_o_mask
		if non_o_mask.sum() != 0:
			return float(overlap.sum().item()) / non_o_mask.sum().item()
		else:
			return 1.0


	def record_log(self, log_name, label_names, class_names, flat_v_label, flat_pred_idx, flat_role_label):
		batch_l = self.shared.batch_l
		orig_l = self.shared.orig_seq_l
		v_label = self.shared.v_label
		v_l = self.shared.v_l
		#bv_idx = int(np.where(self.labels == 'B-V')[0][0])

		#frame_idx = self.shared.res_map['frame']	# batch_l, source_l
		#frame_pool = self.shared.res_map['frame_pool']	# num_prop, num_frame, num_label

		start = 0
		for i in range(batch_l):
			# do analysis without cls and sep
			orig_tok_grouped = self.shared.res_map['orig_tok_grouped'][i]
			sent = ' '.join(orig_tok_grouped).lower()
			if self.sent_filter and sent not in self.sent_filter:
				self.filtered_sent.add(sent)
				start += v_l[i]
				continue

			orig_l_i = orig_l[i].item()	# convert to scalar
			v_i = v_label[i, :v_l[i]]
			end = start + v_l[i]

			role_i = flat_role_label[start:end, :orig_l_i]
			a_gold_i = torch.zeros(orig_l_i, orig_l_i).long()	# O has idx 0
			for k, role_k in enumerate(role_i):
				a_gold_i[v_i[k]] = role_k

			a_pred_i = flat_pred_idx[start:end, :orig_l_i].cpu()
			# only keep predictions on the predicate positions
			a_pred_i_new = torch.zeros(orig_l_i, orig_l_i).long()	# O has idx 0
			for k in range(v_l[i]):
				a_pred_i_new[v_i[k]] = a_pred_i[k]
				# force it to be the vn class
				# TODO, double check the coverage is correct
				if self.opt.use_gold_predicate == 1:
					a_pred_i_new[v_i[k], v_i[k]] = role_i[k, v_i[k]]
			a_pred_i = a_pred_i_new

			self.gold_log[log_name].append(self.compose_log(label_names, orig_tok_grouped[1:-1], a_gold_i[1:-1, 1:-1]))
			self.pred_log[log_name].append(self.compose_log(label_names, orig_tok_grouped[1:-1], a_pred_i[1:-1, 1:-1]))
			self.pretty_gold[log_name].append(self.compose_log(label_names, orig_tok_grouped[1:-1], a_gold_i[1:-1, 1:-1], transpose=False))
			self.pretty_pred[log_name].append(self.compose_log(label_names, orig_tok_grouped[1:-1], a_pred_i[1:-1, 1:-1], transpose=False))
			self.orig_toks[log_name].append(orig_tok_grouped[1:-1])
			for k in range(v_l[i]):
				v_quick_f1 = self._quick_acc(flat_pred_idx[start + k], flat_role_label[start + k])
				self.v_quick_f1[log_name].append(f'{class_names[flat_v_label[start + k]]}\t{v_quick_f1}')

			start += v_l[i]


	# return a string of stats
	def print_cur_stats(self):
		acc = self.quick_acc_sum / self.num_prop if self.num_prop else 0.0
		stats = 'Quick acc {0:.3f}'.format(acc)
		return stats

	# get training metric (scalar metric, extra metric)
	def get_epoch_metric(self):
		if self.inconsistent_bio_cnt != 0:
			print(self.name + ' inconsistent_bio_cnt: ', self.inconsistent_bio_cnt)
		if self.shared.is_train:
			quick_acc = self.quick_acc_sum / self.num_prop if self.num_prop else 0.0
			return quick_acc, [quick_acc]
		else:
			f1 = self.eval_conll_f1(self.log_name, self.opt.conll_output + f'.{self.log_name}')
			return f1, [f1]

	# compose log for one example
	#	role_labels of shape (seq_l, seq_l)
	def compose_log(self, label_names, orig_toks, role_labels, transpose=True):
		role_labels = role_labels.numpy()
		seq_l = role_labels.shape[0]
#
		header = ['-' for _ in range(seq_l)]
		role_lines = []
		for i, row in enumerate(role_labels):
			roles = label_names[row].tolist()
			roles = roles + ['O']	# TODO, the convert_role_labels prefers the last label to be O, so bit hacky here
			if is_v(roles[i]):	# on the diagonal, if the i-th label is a predicate, then tag it
				header[i] = orig_toks[i]
				roles, error_cnt = convert_role_labels(roles)
				role_lines.append(roles[:-1])
				self.inconsistent_bio_cnt += error_cnt
#
		log = [header] + role_lines
		log = np.asarray(log)
		# do a transpose
		if transpose:
			log = log.transpose((1, 0))
		
		rs = []
		for row in log:
			rs.append(' '.join(row))
		return '\n'.join(rs) + '\n'

	def eval_conll_f1(self, log_name, path):
		# eval vn pred
		print('writing gold to {}'.format(f'{path}_gold.txt'))
		with open(f'{path}_gold.txt', 'w') as f:
			for ex in self.gold_log[log_name]:
				f.write(ex + '\n')
	
		print('writing pred to {}'.format(f'{path}_pred.txt'))
		with open(f'{path}_pred.txt', 'w') as f:
			for ex in self.pred_log[log_name]:
				f.write(ex + '\n')

		return system_call_eval(f'{path}_gold.txt', f'{path}_pred.txt')

	def begin_pass(self):
		# clear stats
		self.quick_acc_sum = 0
		self.quick_v_acc_sum = 0
		self.num_ex = 0
		self.num_prop = 0
		self.gold_log = {self.log_name: []}
		self.pred_log = {self.log_name: []}
		self.pretty_gold = {self.log_name: []}
		self.pretty_pred = {self.log_name: []}
		self.orig_toks = {self.log_name: []}
		self.v_quick_f1 = {self.log_name: []}
		self.inconsistent_bio_cnt = 0
		self.conf_map = torch.zeros(len(self.label_groups), len(self.label_groups))
		self.filtered_sent = set()
		#self.frame_log = []

	def end_pass(self):
		if 'pretty' in self.log_types:
			print('writing pretty gold to {}'.format(self.opt.conll_output + f'.{self.name}_{self.log_name}_pretty_gold.txt'))
			with open(self.opt.conll_output + f'.{self.name}_{self.log_name}_pretty_gold.txt', 'w') as f:
				for toks, ex in zip(self.orig_toks[self.log_name], self.pretty_gold[self.log_name]):
					f.write(' '.join(toks)+'\n')
					f.write(ex + '\n')
#
			print('writing pretty pred to {}'.format(self.opt.conll_output + f'.{self.name}_{self.log_name}_pretty_pred.txt'))
			with open(self.opt.conll_output + f'.{self.name}_{self.log_name}_pretty_pred.txt', 'w') as f:
				for toks, ex in zip(self.orig_toks[self.log_name], self.pretty_pred[self.log_name]):
					f.write(' '.join(toks)+'\n')
					f.write(ex + '\n')

			print('writing v quick f1 to {}'.format(self.opt.conll_output + f'.{self.name}_{self.log_name}_v_quick_f1.txt'))
			with open(self.opt.conll_output + f'.{self.name}_{self.log_name}_v_quick_f1.txt', 'w') as f:
				for line in self.v_quick_f1[self.log_name]:
					f.write(line + '\n')
#
			print(f'number of sentences {self.num_ex - len(self.filtered_sent)}, and number in filter {len(self.sent_filter)}')

if __name__ == '__main__':
	pass
