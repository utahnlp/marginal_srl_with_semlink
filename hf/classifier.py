import torch
from torch import nn
from util.holder import *
from util.util import *

class MergeConcatClassifier(torch.nn.Module):
	def __init__(self, opt, shared):
		super(MergeConcatClassifier, self).__init__()
		self.opt = opt
		self.shared = shared

		self.num_condensed_label = len(self.opt.condensed_labels)

		self.vn_labels = np.asarray(self.opt.vn_labels)
		self.srl_labels = np.asarray(self.opt.srl_labels)
		self.cross2condensed = torch.tensor(self.opt.cross2condensed, dtype=torch.long)
		self.cross2condensed = to_device(self.cross2condensed, self.opt.gpuid)

		# transformation to get phi_vs(x)
		self.vn_f_v = nn.Sequential(
			nn.Linear(opt.hidden_size, opt.hidden_size))

		self.vn_f_a = nn.Sequential(
			nn.Linear(opt.hidden_size, opt.hidden_size))

		self.vn_g_va = nn.Sequential(
			#nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size*2, opt.hidden_size),
			nn.GELU(),
			#nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.hidden_size),
			nn.GELU())

		self.srl_f_v = nn.Sequential(
			nn.Linear(opt.hidden_size, opt.hidden_size))

		self.srl_f_a = nn.Sequential(
			nn.Linear(opt.hidden_size, opt.hidden_size))

		self.srl_g_va = nn.Sequential(
			#nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size*2, opt.hidden_size),
			nn.GELU(),
			#nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.hidden_size),
			nn.GELU())

		self.v_layer = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.num_vn_label))

		self.vn_srl_label_layer = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size*2, opt.hidden_size),
			nn.GELU(),
			nn.Linear(opt.hidden_size, self.num_condensed_label))


	def extract_v_enc(self, enc):
		assert(len(enc.shape) == 3)
		batch_l, source_l, _ = enc.shape
		v_l = self.shared.v_l
		v_label = self.shared.v_label

		flat_enc = []
		start = 0
		batch_v_idx = []
		for i in range(batch_l):
			enc_i = enc[i].index_select(0, v_label[i, :v_l[i]])
			flat_enc.append(enc_i)
			start += v_l[i]
			batch_v_idx.extend(((p, q) for p, q in zip([i]*v_l[i], v_label[i, :v_l[i]])))
		flat_enc = torch.cat(flat_enc, dim=0)
		return flat_enc, batch_v_idx


	def extract_a_enc(self, enc):
		assert(len(enc.shape) == 3)
		batch_l, source_l, hidden_size = enc.shape
		v_l = self.shared.v_l
		v_label = self.shared.v_label

		flat_enc = []
		start = 0
		for i in range(batch_l):
			enc_i = enc[i].view(1, source_l, -1)
			flat_enc.append(enc_i.expand(v_l[i], source_l, hidden_size))
			start += v_l[i]
		flat_enc = torch.cat(flat_enc, dim=0)
		return flat_enc


	def forward(self, enc):
		vn_enc = compact_subtoks(enc, self.shared.sub2tok_idx, self.opt.compact_mode)
		srl_enc = vn_enc

		(batch_l, source_l, hidden_size) = vn_enc.shape

		# v predictor
		v_score = self.v_layer(vn_enc.view(-1, hidden_size)).view(batch_l, source_l, self.opt.num_vn_label)
		log_v = nn.LogSoftmax(-1)(v_score)

		# pass for vn
		v_enc, batch_v_idx = self.extract_v_enc(vn_enc)	# (num_v, hidden_size)
		a_enc = self.extract_a_enc(vn_enc)	# (num_v, source_l, hidden_size)
		num_v = v_enc.shape[0]
		v_enc = self.vn_f_v(v_enc).view(num_v, 1, self.opt.hidden_size)
		a_enc = self.vn_f_a(a_enc).view(num_v, source_l, self.opt.hidden_size)
		# forming v-a pairing tensor
		va_enc = torch.cat([
			v_enc.expand(num_v, source_l, self.opt.hidden_size),
			a_enc.expand(num_v, source_l, self.opt.hidden_size)], dim=-1)
		va_enc = self.vn_g_va(va_enc.view(-1, self.opt.hidden_size*2))
		vn_final = va_enc.view(num_v, source_l, self.opt.hidden_size)

		self.shared.batch_v_idx = batch_v_idx

		# pass for srl
		v_enc, _ = self.extract_v_enc(srl_enc)	# (num_v, hidden_size)
		a_enc = self.extract_a_enc(srl_enc)	# (num_v, source_l, hidden_size)
		v_enc = self.srl_f_v(v_enc).view(num_v, 1, self.opt.hidden_size)
		a_enc = self.srl_f_a(a_enc).view(num_v, source_l, self.opt.hidden_size)
		# forming a large tensor
		va_enc = torch.cat([
			v_enc.expand(num_v, source_l, self.opt.hidden_size),
			a_enc.expand(num_v, source_l, self.opt.hidden_size)], dim=-1)
		va_enc = self.srl_g_va(va_enc.view(-1, self.opt.hidden_size*2))
		srl_final = va_enc.view(num_v, source_l, self.opt.hidden_size)

		# concat to run through the vn_srl_label_layer
		concated = torch.concat([vn_final, srl_final], dim=-1)	# (num_v, source_l, hidden_size*2)
		vn_srl_score = self.vn_srl_label_layer(concated.view(-1, self.opt.hidden_size*2)).view(num_v, source_l, self.num_condensed_label)
		log_vn_srl = nn.LogSoftmax(-1)(vn_srl_score)

		# get vn and srl label scores as well, by merging scores from the condensed list
		cross2condensed_idx = self.cross2condensed.view(1, -1).expand(num_v*source_l, -1)	# (num_v * source_l, num_vn * num_srl)
		condensed_score = vn_srl_score.view(num_v*source_l, -1, 1)	# (num_v * source_l, num_condensed, 1)	last 1 so we can use batch_index1_select
		cross_score = batch_index1_select(condensed_score, cross2condensed_idx, nul_idx=-1).view(
			num_v, source_l, self.opt.num_vn_label, self.opt.num_srl_label)
		vn_score = cross_score.max(-1)[0]
		srl_score = cross_score.max(-2)[0]
		log_vn = nn.LogSoftmax(-1)(vn_score)
		log_srl = nn.LogSoftmax(-1)(srl_score)


		pack = {'log_score': log_vn_srl, 'label_score': vn_srl_score,
				'log_srl': log_srl, 'srl_score': srl_score,
				'log_vn': log_vn, 'vn_score': vn_score}

		return pack


	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	pass





		