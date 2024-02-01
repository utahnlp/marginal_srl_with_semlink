import sys
import torch
from torch import nn
from torch.autograd import Variable
from util.holder import *
from util.util import *
from .crf import *
from collections import defaultdict

# Cross CRF loss that does cross product of VN and SRL labels and condense them into one CRF
class CrossCRFLoss(torch.nn.Module):
	def __init__(self, name, opt, shared):
		super(CrossCRFLoss, self).__init__()
		self.opt = opt
		self.shared = shared
		self.name = name

		self.vn_labels = np.asarray(self.opt.vn_labels)
		self.srl_labels = np.asarray(self.opt.srl_labels)
		self.condensed_labels = np.asarray(self.opt.condensed_labels)
		self.cross2condensed = torch.tensor(self.opt.cross2condensed, dtype=torch.long)
		self.cross2condensed = to_device(self.cross2condensed, self.opt.gpuid)
		self.condensed2cross = torch.tensor(self.opt.condensed2cross, dtype=torch.long)
		self.condensed2cross = to_device(self.condensed2cross, self.opt.gpuid)
		self.condensed2srl = torch.tensor(self.opt.condensed2srl, dtype=torch.long)
		self.condensed2srl = to_device(self.condensed2srl, self.opt.gpuid)

		self.vn_classes = np.asarray(self.opt.vn_classes)
		self.roleset_suffix = np.asarray(self.opt.roleset_suffix)
		self.roleset = np.asarray(self.opt.roleset)

		# only mark the B-* labels
		self.condensed_b_mask = [1 if l.startswith('B-') else 0 for l in self.opt.condensed_labels]
		self.condensed_b_mask = to_device(torch.Tensor(self.condensed_b_mask), self.opt.gpuid)

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
		self.condensed_content_mask = to_device(torch.tensor(self.condensed_content_mask), self.opt.gpuid)
		self.condensed_core_masks = to_device(torch.tensor(self.condensed_core_masks), self.opt.gpuid)

		self.srl_core_masks = [[0 for _ in self.opt.srl_labels] for _ in range(6)] 	# ARG0-ARG5
		self.srl_core_label_idx = []
		for i, srl_l in enumerate(self.opt.srl_labels):
			if not srl_l.startswith('B-'):
				continue
			for c in range(6):
				if f'ARG{c}' in srl_l:
					self.srl_core_masks[c][i] = 1
					self.srl_core_label_idx.append(i)
		self.srl_core_masks = to_device(torch.tensor(self.srl_core_masks), self.opt.gpuid)
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
		self.vn2condensed_mask = to_device(self.vn2condensed_mask, self.opt.gpuid)
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
		self.srl2condensed_mask = to_device(self.srl2condensed_mask, self.opt.gpuid)
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
		self.vn_b2i = to_device(torch.tensor(self.vn_b2i), self.opt.gpuid)
		self.srl_b2i = to_device(torch.tensor(self.srl_b2i), self.opt.gpuid)


		# make CRF module
		constraints = allowed_transitions("BIO", self.opt.condensed_map_inv)
		self.crf = ConditionalRandomField(num_tags=len(self.condensed_labels), constraints=constraints, gpuid=opt.gpuid)

		self.quick_acc_sum = 0.0
		self.num_ex = 0
		self.num_prop = 0
		self.inconsistent_bio_cnt = 0
		self.semlink_gold_vio_cnt = 0
		self.semlink_pred_vio_cnt = 0

		self.log_types = []
		if hasattr(self.opt, 'logs'):
			self.log_types = self.opt.logs.strip().split(',')


	def map_cross_to_condensed(self, vn_seq, srl_seq):
		zero = to_device(torch.tensor(0, dtype=torch.long), self.opt.gpuid)
		role_label = self.cross2condensed[vn_seq, srl_seq]
		# replace missed entries (-1) with O
		return torch.where(role_label != -1, role_label, zero)


	def map_condensed_to_cross(self, seq):
		pairs = self.condensed2cross[seq]
		return pairs[..., 0], pairs[..., 1]


	def analyze_frame(self, srl_pred):
		srl_pred = srl_pred.cpu()
		batch_l = self.shared.batch_l
		v_l = self.shared.v_l
		orig_l = self.shared.orig_seq_l
		frame_pool = self.shared.frame_pool.cpu()
		frame_idx = self.shared.frame_idx.cpu()
		roleset_suffix = self.shared.roleset_suffixes[:, :v_l.max()]

		start = 0
		vio_cnt = 0
		for i in range(batch_l):
			for k in range(v_l[i]):
				lemma_idx = frame_idx[i, k]
				suffix_idx = roleset_suffix[i, k]
				assert(lemma_idx > 0 and suffix_idx > 0)

				arg_mask = frame_pool[lemma_idx, suffix_idx]
				assert(len(arg_mask) <= self.opt.num_srl_label)

				mask = torch.zeros(self.opt.num_srl_label)
				mask[: len(arg_mask)] = arg_mask

				invalid_mask = 1.0 - mask
				role_pred = srl_pred[start, :orig_l[i]]
				invalid_pred = invalid_mask.unsqueeze(0).expand(orig_l[i], self.opt.num_srl_label).gather(-1, role_pred.unsqueeze(-1))	# (orig_l, 1)
				if invalid_pred.sum() > 0:
					vio_cnt += 1
				start += 1
		assert(start == srl_pred.shape[0])
		return vio_cnt


	def get_frame_penalty_mask(self, score):
		v_l = self.shared.v_l
		num_v, orig_l, condensed_l = score.shape
		frame_pool = self.shared.frame_pool
		frame_idx = self.shared.frame_idx
		roleset_suffix = self.shared.roleset_suffixes[:, :v_l.max()]
		core_b_mask = self.condensed_core_masks.any(0).logical_and(self.condensed_b_mask)

		lemma_idx = []
		suffix_idx = []
		for i in range(self.shared.batch_l):
			lemma_idx.extend(frame_idx[i, :v_l[i]])
			suffix_idx.extend(roleset_suffix[i, :v_l[i]])
		lemma_idx = torch.tensor(lemma_idx, dtype=torch.long).to(score.device)
		suffix_idx = torch.tensor(suffix_idx, dtype=torch.long).to(score.device)

		assert(lemma_idx.shape == (num_v,) and suffix_idx.shape == lemma_idx.shape)

		# num_v, num_suffix, num_srl_label
		pool = frame_pool[lemma_idx]
		# num_v, num_srl_label
		frame_mask = batch_index1_select(pool, suffix_idx.view(num_v, 1), nul_idx=0).squeeze(1)
		# Might be inconsistent due to the order of data processing
		if frame_mask.shape[1] < self.opt.num_srl_label:
			mask = torch.zeros(num_v, self.opt.num_srl_label).to(frame_mask.device)
			mask[:, :frame_mask.shape[1]] = frame_mask
			frame_mask = mask

		frame_mask = frame_mask.float()
		# num_v, num_srl_label, condensed_l
		srl_beam_mask = frame_mask.view(num_v, self.opt.num_srl_label, 1) * self.srl2condensed_mask.unsqueeze(0)
		# num_v, condensed_l
		srl_beam_mask = srl_beam_mask.any(1)
		# num_v, condensed_l
		frame_invalid_core_mask = srl_beam_mask.logical_not().logical_and(core_b_mask.view(1, -1))	# condensed_l

		# return score + frame_invalid_core_mask.unsqueeze(1) * self.opt.neg_inf
		return frame_invalid_core_mask	# (num_v, condensed_l)


	def pretty_print_frame(self):
		batch_l = self.shared.batch_l
		v_l = self.shared.v_l
		v_label = self.shared.v_label.cpu()
		frame_pool = self.shared.frame_pool.cpu()
		frame_idx = self.shared.frame_idx.cpu()

		roleset_suffix = self.shared.roleset_suffixes[:, :v_l.max()]

		rs = []
		for i in range(batch_l):
			log = []
			orig_toks = orig_tok_grouped = self.shared.res_map['orig_tok_grouped'][i][1:-1]
			for k in range(v_l[i]):
				lemma_idx = frame_idx[i, k]
				suffix_idx = roleset_suffix[i, k]
				assert(lemma_idx > 0 and suffix_idx > 0)

				arg_mask = frame_pool[lemma_idx, suffix_idx]
				assert(len(arg_mask) <= self.opt.num_srl_label)

				mask = torch.zeros(self.opt.num_srl_label)
				mask[: len(arg_mask)] = arg_mask

				args = [l for m, l in zip(mask, self.opt.srl_labels) if m]
				args = sorted(list(set([parse_label(p)[1] for p in args])))
				line = f'frame: {",".join(args)}'
				log.append(line)
			rs.append('\n'.join(log) + '\n')
		return rs


	def pretty_print_semlink(self):
		batch_l = self.shared.batch_l
		v_l = self.shared.v_l
		v_label = self.shared.v_label.cpu()
		semlink_l = self.shared.semlink_l.cpu()
		semlink = self.shared.semlink[:, :v_l.max(), :, :semlink_l.max()].cpu()

		vn_class = self.shared.vn_class[:, :v_l.max()]
		roleset_suffix = self.shared.roleset_suffixes[:, :v_l.max()]

		rs = []
		for i in range(batch_l):
			log = []
			orig_toks = orig_tok_grouped = self.shared.res_map['orig_tok_grouped'][i][1:-1]
			for k in range(v_l[i]):
				vnc = vn_class[i, k]
				suffix = roleset_suffix[i, k]

				mapping = semlink[i, k]
				mapping = list((self.srl_labels[mapping[0, k]], self.vn_labels[mapping[1, k]]) for k in range(mapping.shape[1])
					if mapping[0, k] != -1 and mapping[1, k] != -1)
				# a bit of clean up
				mapping = list(set((p[0][2:], p[1][2:]) for p in mapping))
				mapping = list(p for p in mapping if not p[0].startswith('R-') and not p[0].startswith('C-'))
				mapping = [(self.opt.vn_classes[vnc], self.opt.roleset_suffix[suffix])] + mapping
				log_k = ','.join(f'{p[0]}~{p[1]}' for p in mapping)
				log.append(log_k)
			# print('\n'.join(log) + '\n')
			rs.append('\n'.join(log) + '\n')
		return rs


	def analyze_semlink(self, vn_pred, srl_pred):
		vn_pred = vn_pred.cpu()
		srl_pred = srl_pred.cpu()
		batch_l = self.shared.batch_l
		v_l = self.shared.v_l
		semlink_l = self.shared.semlink_l.cpu()
		semlink = self.shared.semlink[:, :v_l.max(), :, :semlink_l.max()].cpu()

		vio_cnt = 0
		start = 0
		for i in range(batch_l):
			orig_tok_grouped = self.shared.res_map['orig_tok_grouped'][i]
			for k in range(v_l[i]):
				vn = vn_pred[start]
				srl = srl_pred[start]
				mapping = semlink[i, k]
				mapping = list((self.srl_labels[mapping[0, z]], self.vn_labels[mapping[1, z]]) for z in range(mapping.shape[1])
					if mapping[0, z] != -1 and mapping[1, z] != -1)
				if not mapping:
					start += 1
					continue
				has_vio = False
				for a, r in zip(vn, srl):
					if a <= 0 or r <= 0:
						continue
					r_label = self.srl_labels[r]
					a_label = self.vn_labels[a]
					if r_label == 'B-V' or a_label == 'B-V':
						continue
					if r_label.startswith('B-') or a_label.startswith('B-'):
						if (r_label, a_label) not in mapping:
							vnc_id = self.shared.vn_class[i, k].item()
							roleset_suffix_id = self.shared.roleset_suffixes[i, k].item()
							print(self.opt.vn_class_map_inv[vnc_id], self.opt.roleset_suffix_map_inv[roleset_suffix_id])
							print(mapping)
							print(r_label, a_label)
							has_vio = True
							break
				if has_vio:
					vio_cnt += 1
				start += 1
		return vio_cnt


	def get_semlink_penalty_mask(self, a_semlink):
		zero = to_device(torch.tensor(0, dtype=torch.long), self.opt.gpuid)
		num_v = a_semlink.shape[0]
		condensed_l = len(self.opt.condensed_labels)
		assert(len(a_semlink.shape) == 3)	# (num_v, 2, max_num_semlink)
		# importantly, replace -1 with index that points to O (ie 0)
		b_roles = a_semlink[:, 0, :]
		b_roles = torch.where(b_roles != -1, b_roles, zero)	# (num_v, max_num_semlink)
		i_roles = self.srl_b2i[b_roles]
		b_args = a_semlink[:, 1, :]
		b_args = torch.where(b_args != -1, b_args, zero)		# (num_v, max_num_semlink)
		i_args = self.vn_b2i[b_args]
		# (num_v, max_num_semlink*2)
		roles = torch.cat([b_roles, i_roles], dim=-1)
		args = torch.cat([b_args, i_args], dim=-1)
		# (num_v, max_num_semlink*2, condensed_l)
		srl_mask = self.srl2condensed_mask[roles]
		vn_mask = self.vn2condensed_mask[args]

		# Inner mask denotes the explicit content label pairs mentioned in semlink.
		inner_mask = torch.logical_and(srl_mask, vn_mask)
		# each of the semlink corresponds to exactly one label in the condensed list.
		assert(torch.all(inner_mask.int().sum(-1) <= 1))
		# (num_v, condensed_l)
		inner_mask = inner_mask.logical_and(self.condensed_content_mask.view(1, 1, condensed_l))
		inner_mask = inner_mask.any(1)

		# What we want is that content label pairs not in the inner mask are disabled, e.g.,
		# 	if semlink defines ARG0-AGENT and ARG1-THEME, then ARG0-non-AGENT, ARG1-non-THEME, non-ARG0-AGENT, and non-ARG1-THEME are blocked.
		#	ARG2 is only allowed to O
		disable_mask = inner_mask.logical_not().logical_and(self.condensed_content_mask.view(1, condensed_l))
		# Furthermore, if the incoming semlink is empty, we disable nothing
		valid_semlink = (roles == 0).all(-1).logical_not().view(num_v, 1)
		disable_mask = disable_mask.logical_and(valid_semlink)

		# (num_v, condensed_l)
		return disable_mask


	def gen_unique_core_violation(self, gold_srl, num_sample=1):
		srl_core_mask = self.srl_core_masks.any(0)	# (num_srl,)
		# (num_v, orig_l)
		is_core = srl_core_mask[gold_srl]
		# (num_v, orig_l)
		gold_core = torch.where(is_core, gold_srl, 0)
		# make random duplicate
		dup_srl = []
		for _ in range(num_sample):
			dup_core = random_duplicate(gold_core)
			# recover non-core labels
			dup_core = torch.where(is_core, dup_core, gold_srl)
			dup_srl.append(dup_core)
		return dup_srl


	def analyze_srl_unique(self, srl_idx):
		num_v, orig_l = srl_idx.shape
		v_vio_mask = torch.zeros((num_v,), dtype=torch.bool, device=srl_idx.device)
		for k, core in enumerate(self.srl_core_label_idx):
			has_dup = ((srl_idx == core).sum(-1) > 1)	# (num_v)
			v_vio_mask = v_vio_mask.logical_or(has_dup)
		return v_vio_mask.sum().item()


	def decode(self, pack, v_label, v_l):
		flat_score = pack['label_score']

		num_v = flat_score.shape[0]
		orig_l = self.shared.orig_seq_l
		max_orig_l = orig_l.max()
		semlink_l = self.shared.semlink_l
		semlink = self.shared.semlink[:, :v_l.max(), :, :semlink_l.max()]

		v_label = to_device(v_label, self.opt.gpuid)
		v_l = to_device(v_l, self.opt.gpuid)

		# pack everything into (num_v, max_orig_l, ...)
		flat_mask = []
		flat_semlink = []
		for i in range(batch_l):
			mask_i = torch.zeros(v_l[i], max_orig_l).byte()
			mask_i[:, :orig_l[i]] = True
			flat_mask.append(mask_i)
			flat_semlink.append(semlink[i, :v_l[i]])

		flat_score = flat_score[:, :max_orig_l]
		flat_mask = to_device(torch.cat(flat_mask, dim=0), self.opt.gpuid)
		flat_semlink = to_device(torch.cat(flat_semlink, dim=0), self.opt.gpuid)

		if self.opt.use_semlink_crf == 1:
			semlink_penalty_mask = self.get_semlink_penalty_mask(flat_semlink)
			flat_score += semlink_penalty_mask.unsqueeze(1) * self.opt.neg_inf

		decoded = self.crf.viterbi_tags(flat_score, flat_mask)

		# unpack decoded to two (batch_l, max_orig_l, max_orig_l) tensors
		# one for vn, one for srl.
		flat_vn_pred = torch.zeros(num_v, max_orig_l).long()
		flat_srl_pred = torch.zeros(num_v, max_orig_l).long()
		start = 0
		for i in range(batch_l):
			for k in range(v_l[i]):
				p = torch.Tensor(decoded[start][0]).long()
				vn_seq, srl_seq = self.map_condensed_to_cross(p)

				flat_vn_pred[start, :orig_l[i]] = vn_seq
				flat_srl_pred[start, :orig_l[i]] = srl_seq
				start += 1
		assert(start == len(decoded))
		flat_vn_pred = to_device(flat_vn_pred, self.opt.gpuid)
		flat_srl_pred = to_device(flat_srl_pred, self.opt.gpuid)

		# a bit of analyze
		self.semlink_pred_vio_cnt += self.analyze_semlink(flat_vn_pred, flat_srl_pred)

		return flat_vn_pred, flat_vn_pred


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

		# role_label is in condensed mode
		condensed_label = self.map_cross_to_condensed(self.shared.vn_label, self.shared.srl_label)
		srl_label = self.shared.srl_label

		flat_semlink = []
		flat_gold = []
		flat_mask = []
		flat_gold_srl = []
		# pack everything into (num_v, max_orig_l, ...)
		for i in range(batch_l):
			gold_i = torch.zeros(v_l[i], max_orig_l).long()	# O has idx 0
			gold_i[:, :orig_l[i]] = condensed_label[i, :v_l[i], :orig_l[i]]	# (num_v, orig_l)
			gold_srl_i = torch.zeros(v_l[i], max_orig_l).long()	# O has idx 0
			gold_srl_i[:, :orig_l[i]] = srl_label[i, :v_l[i], :orig_l[i]]	# (num_v, orig_l)
			mask_i = torch.zeros(v_l[i], max_orig_l).byte()
			mask_i[:, :orig_l[i]] = True

			flat_mask.append(mask_i)
			flat_gold.append(gold_i)
			flat_gold_srl.append(gold_srl_i)
			flat_semlink.append(semlink[i, :v_l[i]])

		flat_score = flat_score[:, :max_orig_l]
		flat_mask = to_device(torch.cat(flat_mask, dim=0), self.opt.gpuid)
		flat_gold = to_device(torch.cat(flat_gold, dim=0), self.opt.gpuid)
		flat_gold_srl = to_device(torch.cat(flat_gold_srl, dim=0), self.opt.gpuid)
		flat_semlink = to_device(torch.cat(flat_semlink, dim=0), self.opt.gpuid)
		flat_pred = flat_log_score[:, :max_orig_l].argmax(-1)

		if self.shared.is_train:
			if 'cross_marginal_srl' == self.shared.batch_flag:
				# (num_v, seq_l, condensed_l)
				flat_beam_mask = self.srl2condensed_mask[flat_gold_srl]
				loss = self.crf.marginal(flat_score, flat_beam_mask, flat_mask)
				loss *= self.opt.marginal_w
			elif 'cross_marginal_srl_semlink' == self.shared.batch_flag:
				semlink_penalty_mask = self.get_semlink_penalty_mask(flat_semlink)
				semlink_penalty_mask = semlink_penalty_mask.unsqueeze(1)	# (num_v, 1, condensed_l)
				flat_beam_mask = self.srl2condensed_mask[flat_gold_srl]		# (num_v, seq_l, condensed_l)
				flat_beam_mask = flat_beam_mask.logical_and(semlink_penalty_mask.logical_not())
				loss = self.crf.marginal(flat_score, flat_beam_mask, flat_mask)
				loss *= self.opt.marginal_w
			elif 'cross_marginal_srl_semlink_local' == self.shared.batch_flag:
				semlink_penalty_mask = self.get_semlink_penalty_mask(flat_semlink)
				semlink_penalty_mask = semlink_penalty_mask.unsqueeze(1)	# (num_v, 1, condensed_l)
				flat_srl_mask = self.srl2condensed_mask[flat_gold_srl]		# (num_v, seq_l, condensed_l)
				flat_score += semlink_penalty_mask * self.opt.neg_inf
				loss = self.crf.marginal(flat_score, flat_srl_mask, flat_mask)
				loss *= self.opt.marginal_w
			elif 'cross_marginal_srl_unique' == self.shared.batch_flag:
				# (num_v, seq_l, condensed_l)
				flat_beam_mask = self.srl2condensed_mask[flat_gold_srl]
				log_srl = self.crf._marginal_likelihood(flat_score, flat_beam_mask, flat_mask)
				log_Z = self.crf._input_likelihood(flat_score, flat_mask)
				# log of the marginalized probability of srl-only over the partition
				log_p_srl = log_srl - log_Z

				log_not_p_dup = 0
				for flat_dup_srl in self.gen_unique_core_violation(flat_gold_srl,num_sample=self.opt.srl_unique_sample):
					flat_dup_beam_mask = self.srl2condensed_mask[flat_dup_srl]
					# log of the "marginalized" probability of duplicate core label samples over the partition
					log_dup = self.crf._marginal_likelihood(flat_score, flat_dup_beam_mask, flat_mask)
					log_p_dup = log_dup - log_Z
					log_not_p_dup = (1 - log_p_dup.exp()).clamp(min=1e-3).log() + log_not_p_dup

				loss = - torch.sum(log_p_dup + log_not_p_dup)
				loss *= self.opt.marginal_w
			elif 'cross_semlink' == self.shared.batch_flag:
				semlink_penalty_mask = self.get_semlink_penalty_mask(flat_semlink)
				semlink_penalty_mask = semlink_penalty_mask.unsqueeze(1)	# (num_v, 1, condensed_l)
				flat_score += semlink_penalty_mask * self.opt.neg_inf
				loss = self.crf(flat_score, flat_gold, flat_mask)
			else:
				loss = self.crf(flat_score, flat_gold, flat_mask)
			flat_vn_pred, flat_srl_pred = self.map_condensed_to_cross(flat_pred)
			flat_vn_pred = to_device(flat_vn_pred, self.opt.gpuid)
			flat_srl_pred = to_device(flat_srl_pred, self.opt.gpuid)

			self.loss_sum[self.shared.batch_flag] += loss.item()
			self.loss_denom[self.shared.batch_flag] += num_v

		else:
			if self.opt.eval_with_srl == 1:
				# (num_v, seq_l, condensed_l)
				flat_beam_mask = self.srl2condensed_mask[flat_gold_srl]
				flat_score += flat_beam_mask.logical_not() * self.opt.neg_inf

			if self.opt.eval_with_semlink == 1:
				semlink_penalty_mask = self.get_semlink_penalty_mask(flat_semlink)
				flat_score += semlink_penalty_mask.unsqueeze(1) * self.opt.neg_inf

			if self.opt.eval_with_frame == 1:
				frame_penalty_mask = self.get_frame_penalty_mask(flat_score)
				flat_score += frame_penalty_mask.unsqueeze(1) * self.opt.neg_inf

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
			flat_vn_pred = to_device(flat_vn_pred, self.opt.gpuid)
			flat_srl_pred = to_device(flat_srl_pred, self.opt.gpuid)

			flat_vn_gold = torch.zeros(num_v, max_orig_l).long()
			flat_srl_gold = torch.zeros(num_v, max_orig_l).long()
			start = 0
			for i in range(batch_l):
				for k in range(v_l[i]):
					p = flat_gold[start, :orig_l[i]]
					vn_seq, srl_seq = self.map_condensed_to_cross(p)
					flat_vn_gold[start, :orig_l[i]] = vn_seq
					flat_srl_gold[start, :orig_l[i]] = srl_seq
					start += 1
			assert(start == len(decoded))
			flat_vn_gold = to_device(flat_vn_gold, self.opt.gpuid)
			flat_srl_gold = to_device(flat_srl_gold, self.opt.gpuid)
			self.shared.flat_vn_gold = flat_vn_gold
			self.shared.flat_srl_gold = flat_srl_gold

			# in validation mode, no need to count the loss here
			loss = to_device(torch.zeros(1), self.opt.gpuid)

		if not self.shared.is_train:
			# a bit of analyze
			self.semlink_pred_vio_cnt += self.analyze_semlink(flat_vn_pred, flat_srl_pred)

			self.semlink_gold_vio_cnt += self.analyze_semlink(flat_vn_gold, flat_srl_gold)

			self.pretty_semlink.extend(self.pretty_print_semlink())

			# self.pretty_frame.extend(self.pretty_print_frame())

			# self.frame_pred_vio_cnt += self.analyze_frame(flat_srl_pred)

			# self.frame_gold_vio_cnt += self.analyze_frame(flat_srl_gold)

			self.srl_unique_pred_vio_cnt += self.analyze_srl_unique(flat_srl_pred)

			self.srl_unique_gold_vio_cnt += self.analyze_srl_unique(flat_srl_gold)

		# some quick analyses
		batch_acc_sum = self._quick_acc(flat_pred, flat_gold) * num_v
		self.quick_acc_sum += batch_acc_sum
		self.num_ex += batch_l
		self.num_prop += num_v

		if not self.shared.is_train:
			self.record_log('vn', self.vn_labels, self.vn_classes, flat_vn_class, flat_vn_pred, flat_vn_gold)
			self.record_log('srl', self.srl_labels, self.roleset, flat_roleset_ids, flat_srl_pred, flat_srl_gold)
			for i, v_idx in self.shared.batch_v_idx:
				self.ex_v_idx.append((self.shared.batch_ex_idx[i], v_idx))
			self.ex_idx.extend(self.shared.batch_ex_idx)

		self.shared.flat_vn_pred = flat_vn_pred
		self.shared.flat_srl_pred = flat_srl_pred

		# average over number of predicates or num_ex
		return loss / num_v, {
			'flat_vn_pred': flat_vn_pred, 'flat_srl_pred': flat_srl_pred,
			'flat_gold': flat_gold}


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


	def _quick_acc_by_v(self, pred_idx, gold_idx):
		return [self._quick_acc(p, g) for p, g in zip(pred_idx, gold_idx)]


	def record_log(self, log_name, label_names, class_names, flat_v_label, flat_pred_idx, flat_role_label):
		batch_l = self.shared.batch_l
		orig_l = self.shared.orig_seq_l
		v_label = self.shared.v_label
		v_l = self.shared.v_l

		self.quick_f1[log_name].extend(self._quick_acc_by_v(flat_pred_idx, flat_role_label))

		start = 0
		for i in range(batch_l):
			# do analysis without cls and sep
			orig_tok_grouped = self.shared.res_map['orig_tok_grouped'][i]
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
		for k, v in self.loss_sum.items():
			stats += f' {k}:{v / self.loss_denom[k] if self.loss_denom[k] else 0.0}'
		return stats

	# get training metric (scalar metric, extra metric)
	def get_epoch_metric(self):
		if self.inconsistent_bio_cnt != 0:
			print('warning: inconsistent_bio_cnt: ', self.inconsistent_bio_cnt)
		pred_vio = self.semlink_pred_vio_cnt / self.num_prop if self.num_prop else 0.0
		gold_vio = self.semlink_gold_vio_cnt / self.num_prop if self.num_prop else 0.0
		print(f'semlink pred violation: {self.semlink_pred_vio_cnt}/{self.num_prop}, {pred_vio:.4f}')
		print(f'semlink gold violation: {self.semlink_gold_vio_cnt}/{self.num_prop}, {gold_vio:.4f}')

		frame_pred_vio = self.frame_pred_vio_cnt / self.num_prop if self.num_prop else 0.0
		frame_gold_vio = self.frame_gold_vio_cnt / self.num_prop if self.num_prop else 0.0
		print(f'frame pred violation: {self.frame_pred_vio_cnt}/{self.num_prop}, {frame_pred_vio:.4f}')
		print(f'frame gold violation: {self.frame_gold_vio_cnt}/{self.num_prop}, {frame_gold_vio:.4f}')

		srl_unique_pred_vio = self.srl_unique_pred_vio_cnt / self.num_prop if self.num_prop else 0.0
		srl_unique_gold_vio = self.srl_unique_gold_vio_cnt / self.num_prop if self.num_prop else 0.0
		print(f'srl unique pred violation: {self.srl_unique_pred_vio_cnt}/{self.num_prop}, {srl_unique_pred_vio:.4f}')
		print(f'srl unique gold violation: {self.srl_unique_gold_vio_cnt}/{self.num_prop}, {srl_unique_gold_vio:.4f}')

		if self.shared.is_train:
			quick_acc = self.quick_acc_sum / self.num_prop if self.num_prop else 0.0
			return quick_acc, [quick_acc]
		else:
			# eval vn pred
			vn_f1 = self.eval_conll_f1('vn', self.opt.conll_output + f'.{self.name}_vn')
			srl_f1 = self.eval_conll_f1('srl', self.opt.conll_output + f'.{self.name}_srl')
			print('vn f1', vn_f1)
			print('srl f1', srl_f1)
			return (vn_f1 + srl_f1)/2, [vn_f1, srl_f1]

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
				role_lines.append(roles[:-1])	# skip the padded O
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


	def begin_pass(self):
		# clear stats
		self.quick_acc_sum = 0
		self.num_ex = 0
		self.num_prop = 0
		self.gold_log = {'vn': [], 'srl': []}
		self.pred_log = {'vn': [], 'srl': []}
		self.pretty_gold = {'vn': [], 'srl': []}
		self.pretty_pred = {'vn': [], 'srl': []}
		self.orig_toks = {'vn': [], 'srl': []}
		self.quick_f1 = {'vn': [], 'srl': []}
		self.v_quick_f1 = {'vn': [], 'srl': []}
		self.loss_sum = defaultdict(lambda: 0.0)
		self.loss_denom = defaultdict(lambda: 0.0)
		self.ex_v_idx = []
		self.ex_idx = []
		self.pretty_semlink = []
		self.pretty_frame = []
		self.inconsistent_bio_cnt = 0
		self.semlink_pred_vio_cnt = 0
		self.semlink_gold_vio_cnt = 0
		self.frame_pred_vio_cnt = 0
		self.frame_gold_vio_cnt = 0
		self.srl_unique_pred_vio_cnt = 0
		self.srl_unique_gold_vio_cnt = 0
		self.filtered_sent = set()

	def end_pass(self):
		if 'pretty' in self.log_types:
			for log_name in ['vn', 'srl']:
				print('writing pretty gold to {}'.format(self.opt.conll_output + f'.{self.name}_{log_name}_pretty_gold.txt'))
				with open(self.opt.conll_output + f'.{self.name}_{log_name}_pretty_gold.txt', 'w') as f:
					for toks, ex, ex_idx in zip(self.orig_toks[log_name], self.pretty_gold[log_name], self.ex_idx):
						f.write(f'{ex_idx} ||| {" ".join(toks)}\n')
						f.write(ex + '\n')

				print('writing pretty pred to {}'.format(self.opt.conll_output + f'.{self.name}_{log_name}_pretty_pred.txt'))
				with open(self.opt.conll_output + f'.{self.name}_{log_name}_pretty_pred.txt', 'w') as f:
					for toks, ex, ex_idx in zip(self.orig_toks[log_name], self.pretty_pred[log_name], self.ex_idx):
						f.write(f'{ex_idx} ||| {" ".join(toks)}\n')
						f.write(ex + '\n')

			print('writing pretty semlink to {}'.format(self.opt.conll_output + f'.{self.name}_pretty_semlink.txt'))
			with open(self.opt.conll_output + f'.{self.name}_pretty_semlink.txt', 'w') as f:
				for toks, log, ex_idx in zip(self.orig_toks['vn'], self.pretty_semlink, self.ex_idx):
					f.write(f'{ex_idx} ||| {" ".join(toks)}\n')
					f.write(log + '\n')
			
			# print('writing pretty frame to {}'.format(self.opt.conll_output + f'.{self.name}_pretty_frame.txt'))
			# with open(self.opt.conll_output + f'.{self.name}_pretty_frame.txt', 'w') as f:
			# 	for toks, log, ex_idx in zip(self.orig_toks['vn'], self.pretty_frame, self.ex_idx):
			# 		f.write(f'{ex_idx} ||| {" ".join(toks)}\n')
			# 		f.write(log + '\n')

		for log_name in ['vn', 'srl']:
			print('writing quick f1 to {}'.format(self.opt.conll_output + f'.{self.name}_{log_name}_quick_f1.txt'))
			with open(self.opt.conll_output + f'.{self.name}_{log_name}_quick_f1.txt', 'w') as f:
				for f1, ex_v_idx in zip(self.quick_f1[log_name], self.ex_v_idx):
					# f.write(f'{f1} ||| {ex_v_idx[0]},{ex_v_idx[1]-1}\n')		# offset by 1 since we have CLS
					f.write(f'{f1*100}\n')

			print('writing v quick f1 to {}'.format(self.opt.conll_output + f'.{self.name}_{log_name}_v_quick_f1.txt'))
			with open(self.opt.conll_output + f'.{self.name}_{log_name}_v_quick_f1.txt', 'w') as f:
				for line in self.v_quick_f1[log_name]:
					f.write(line + '\n')



if __name__ == '__main__':
	pass




