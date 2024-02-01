import torch
from torch import nn
from util.holder import *
from util.util import *

class VNLinearClassifier(torch.nn.Module):
	def __init__(self, opt, shared):
		super(VNLinearClassifier, self).__init__()
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
			nn.ReLU(),
			#nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.hidden_size),
			nn.ReLU())

		self.label_layer = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.num_vn_label))

		self.v_layer = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.num_vn_label))

		self.class_layer = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.num_vn_class))


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
		a_score = self.label_layer(va_enc.view(-1, self.opt.hidden_size)).view(batch_l, source_l, source_l, self.opt.num_vn_label)
		log_pa = nn.LogSoftmax(-1)(a_score)

		v_score = self.v_layer(enc.view(-1, hidden_size)).view(batch_l, source_l, self.opt.num_vn_label)
		log_v = nn.LogSoftmax(-1)(v_score)

		vnclass_score = self.class_layer(enc.view(-1, hidden_size)).view(batch_l, source_l, self.opt.num_vn_class)
		log_vnclass = nn.LogSoftmax(-1)(vnclass_score)
		pack = {'log_v': log_v, 'v_score': v_score,
			'log_vn': log_pa, 'vn_score': a_score,
			'log_vnclass': log_vnclass, 'vnclass_score': vnclass_score}
		return pack

	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	pass





		