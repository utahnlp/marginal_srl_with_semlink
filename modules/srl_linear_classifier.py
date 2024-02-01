import torch
from torch import nn
from util.holder import *
from util.util import *

class SRLLinearClassifier(torch.nn.Module):
	def __init__(self, opt, shared):
		super(SRLLinearClassifier, self).__init__()
		self.opt = opt
		self.shared = shared

		# transformation to get phi_vs(x)
		self.f_v = nn.Sequential(
			nn.Linear(opt.hidden_size, opt.hidden_size))

		self.f_a = nn.Sequential(
			nn.Linear(opt.hidden_size, opt.hidden_size))

		self.g_va = nn.Sequential(
			#nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size*2, opt.hidden_size),
			nn.GELU(),
			#nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.hidden_size),
			nn.GELU())

		self.label_layer = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.num_srl_label))

		self.v_layer = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.num_srl_label))

		self.sense_layer = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, len(opt.roleset_suffix)))


	def forward(self, enc):
		(batch_l, source_l, hidden_size) = enc.shape
		enc = compact_subtoks(enc, self.shared.sub2tok_idx, self.opt.compact_mode)

		v_enc = self.f_v(enc.view(-1, hidden_size)).view(batch_l, source_l, 1, self.opt.hidden_size)
		a_enc = self.f_a(enc.view(-1, hidden_size)).view(batch_l, 1, source_l, self.opt.hidden_size)
		# forming a large tensor
		va_enc = torch.cat([
			v_enc.expand(batch_l, source_l, source_l, self.opt.hidden_size),
			a_enc.expand(batch_l, source_l, source_l, self.opt.hidden_size)], dim=-1)
		va_enc = self.g_va(va_enc.view(-1, self.opt.hidden_size*2))
		va_enc = va_enc.view(batch_l, source_l, source_l, self.opt.hidden_size)

		# compute score of ijl
		a_score = self.label_layer(va_enc.view(-1, self.opt.hidden_size)).view(batch_l, source_l, source_l, self.opt.num_srl_label)
		log_pa = nn.LogSoftmax(-1)(a_score)

		v_score = self.v_layer(enc.view(-1, hidden_size)).view(batch_l, source_l, self.opt.num_srl_label)
		log_v = nn.LogSoftmax(-1)(v_score)

		sense_score = self.sense_layer(enc.view(-1, hidden_size)).view(batch_l, source_l, len(self.opt.roleset_suffix))
		log_sense = nn.LogSoftmax(-1)(sense_score)
		pack = {'log_v': log_v, 'v_score': v_score,
			'log_sense': log_sense, 'sense_score': sense_score,
			'log_srl': log_pa, 'srl_score': a_score}
		return pack

	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	pass





		