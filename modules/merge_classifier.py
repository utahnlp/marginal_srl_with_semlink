import torch
from torch import nn
from util.holder import *
from util.util import *

class MergeClassifier(torch.nn.Module):
	def __init__(self, opt, shared):
		super(MergeClassifier, self).__init__()
		self.opt = opt
		self.shared = shared

		self.num_condensed_label = len(self.opt.condensed_labels)

		self.vn_labels = np.asarray(self.opt.vn_labels)
		self.srl_labels = np.asarray(self.opt.srl_labels)
		self.condensed2vn = torch.tensor(self.opt.condensed2vn, dtype=torch.long)
		self.condensed2vn = to_device(self.condensed2vn, self.opt.gpuid)
		self.condensed2srl = torch.tensor(self.opt.condensed2srl, dtype=torch.long)
		self.condensed2srl = to_device(self.condensed2srl, self.opt.gpuid)

		assert(self.condensed2vn.shape[-1] == self.condensed2srl.shape[-1]== self.num_condensed_label)

		# transformation to get phi_vs(x)
		self.vn_f_v = nn.Sequential(
			nn.Linear(opt.hidden_size, opt.hidden_size))

		self.vn_f_a = nn.Sequential(
			nn.Linear(opt.hidden_size, opt.hidden_size))

		self.vn_g_va = nn.Sequential(
			#nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size*2, opt.hidden_size),
			nn.ReLU(),
			#nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.hidden_size),
			nn.ReLU())

		self.vn_label_layer = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.num_vn_label))

		self.srl_f_v = nn.Sequential(
			nn.Linear(opt.hidden_size, opt.hidden_size))

		self.srl_f_a = nn.Sequential(
			nn.Linear(opt.hidden_size, opt.hidden_size))

		self.srl_g_va = nn.Sequential(
			#nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size*2, opt.hidden_size),
			nn.ReLU(),
			#nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.hidden_size),
			nn.ReLU())

		self.srl_label_layer = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.num_srl_label))

		self.v_layer = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.num_vn_label))

		self.vn_scale = nn.Sequential(
			nn.Linear(opt.hidden_size*2, opt.hidden_size),
			nn.ReLU(),
			nn.Linear(opt.hidden_size, opt.num_vn_label))

		self.srl_scale = nn.Sequential(
			nn.Linear(opt.hidden_size*2, opt.hidden_size),
			nn.ReLU(),
			nn.Linear(opt.hidden_size, opt.num_srl_label))


	def extract_v_enc(self, enc):
		assert(len(enc.shape) == 3)
		batch_l, source_l, _ = enc.shape
		v_l = self.shared.v_l
		v_label = self.shared.v_label

		flat_enc = []
		start = 0
		for i in range(batch_l):
			enc_i = enc[i].index_select(0, v_label[i, :v_l[i]])
			flat_enc.append(enc_i)
			start += v_l[i]
		flat_enc = torch.cat(flat_enc, dim=0)
		return flat_enc


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
		if self.opt.enc == 'bibert':
			vn_enc, srl_enc = enc
			vn_enc = compact_subtoks(vn_enc, self.shared.sub2tok_idx, self.opt.compact_mode)
			srl_enc = compact_subtoks(srl_vnc, self.shared.sub2tok_idx, self.opt.compact_mode)
		else:
			vn_enc = compact_subtoks(enc, self.shared.sub2tok_idx, self.opt.compact_mode)
			srl_enc = vn_enc

		(batch_l, source_l, hidden_size) = vn_enc.shape

		# v predictor
		v_score = self.v_layer(vn_enc.view(-1, hidden_size)).view(batch_l, source_l, self.opt.num_vn_label)
		log_v = nn.LogSoftmax(-1)(v_score)

		# pass for vn
		v_enc = self.extract_v_enc(vn_enc)	# (num_v, hidden_size)
		a_enc = self.extract_a_enc(vn_enc)	# (num_v, source_l, hidden_size)
		num_v = v_enc.shape[0]
		v_enc = self.vn_f_v(v_enc).view(num_v, 1, self.opt.hidden_size)
		a_enc = self.vn_f_a(a_enc).view(num_v, source_l, self.opt.hidden_size)
		# forming v-a pairing tensor
		va_enc = torch.cat([
			v_enc.expand(num_v, source_l, self.opt.hidden_size),
			a_enc.expand(num_v, source_l, self.opt.hidden_size)], dim=-1)
		va_enc = self.vn_g_va(va_enc.view(-1, self.opt.hidden_size*2))
		va_enc = va_enc.view(num_v, source_l, self.opt.hidden_size)
		# compute score of ijl
		vn_score = self.vn_label_layer(va_enc.view(-1, self.opt.hidden_size)).view(num_v, source_l, self.opt.num_vn_label)
		log_vn = nn.LogSoftmax(-1)(vn_score)
		vn_v_enc = v_enc

		# pass for srl
		v_enc = self.extract_v_enc(srl_enc)	# (num_v, hidden_size)
		a_enc = self.extract_a_enc(srl_enc)	# (num_v, source_l, hidden_size)
		v_enc = self.srl_f_v(v_enc).view(num_v, 1, self.opt.hidden_size)
		a_enc = self.srl_f_a(a_enc).view(num_v, source_l, self.opt.hidden_size)
		# forming a large tensor
		va_enc = torch.cat([
			v_enc.expand(num_v, source_l, self.opt.hidden_size),
			a_enc.expand(num_v, source_l, self.opt.hidden_size)], dim=-1)
		va_enc = self.srl_g_va(va_enc.view(-1, self.opt.hidden_size*2))
		va_enc = va_enc.view(num_v, source_l, self.opt.hidden_size)
		# compute score of ijl
		srl_score = self.srl_label_layer(va_enc.view(-1, self.opt.hidden_size)).view(num_v, source_l, self.opt.num_srl_label)
		log_srl = nn.LogSoftmax(-1)(srl_score)
		srl_v_enc = v_enc

		# merge vn and srl to score condensed label pairs
		# suppose (i,j) denotes a (vn, srl) label pair
		# score(i,j) = vn_scale(i) * vn_part(i) + srl_scale(j) * srl_part(j)
		concated = torch.concat([vn_v_enc, srl_v_enc], dim=-1)	# (num_v, 1, hidden_size*2)
		vn_scale = self.vn_scale(concated.view(-1, self.opt.hidden_size*2)).view(num_v, 1, self.opt.num_vn_label)
		vn_scale = vn_scale.expand(num_v, source_l, self.opt.num_vn_label)
		vn_scale = vn_scale[:, :, self.condensed2vn]
		srl_scale = self.srl_scale(concated.view(-1, self.opt.hidden_size*2)).view(num_v, 1, self.opt.num_srl_label)
		srl_scale = srl_scale.expand(num_v, source_l, self.opt.num_srl_label)
		srl_scale = srl_scale[:, :, self.condensed2srl]

		vn_part = vn_score[:, :, self.condensed2vn]	# (num_v, source_l, num_condesed_label)
		srl_part = srl_score[:, :, self.condensed2srl]
		label_score = vn_scale * vn_part + srl_scale * srl_part
		log_score = nn.LogSoftmax(-1)(label_score)

		pack = {'log_score': log_score, 'label_score': label_score,
				'log_srl': log_srl, 'srl_score': srl_score,
				'log_vn': log_vn, 'vn_score': vn_score}

		return pack


	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	pass





		