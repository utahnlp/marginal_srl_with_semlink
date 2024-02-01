import sys
import torch
from torch import nn
from torch.autograd import Variable
from util.holder import *
from util.util import *
from .crf_loss import *

# Multiple CRF loss function incl. decoding
class MultiCRFLoss(torch.nn.Module):
	def __init__(self, name, opt, shared):
		super(MultiCRFLoss, self).__init__()
		self.opt = opt
		self.shared = shared
		self.name = name

		self.vn_crf = CRFLoss('vn_crf', opt, shared)
		self.srl_crf = CRFLoss('srl_crf', opt, shared)

		self.vn_labels = np.asarray(self.opt.vn_labels)
		self.srl_labels = np.asarray(self.opt.srl_labels)

		self.multi_w = Variable(torch.Tensor([float(p) for p in opt.multi_w.split(',')]), requires_grad=False)
		if self.opt.gpuid != -1:
			self.multi_w = to_device(self.multi_w, self.opt.gpuid)
		assert(len(self.multi_w) == 2)

		self.semlink_gold_vio_cnt = 0
		self.semlink_pred_vio_cnt = 0


	def is_active(self):
		return self.is_vn_active() or self.is_srl_active()


	def is_vn_active(self):
		return 'vn' in self.shared.batch_flag


	def is_srl_active(self):
		return 'srl' in self.shared.batch_flag
 

	# the decode function is for the demo where gold predicate might not present
	def decode(self, pack, v_label=None, v_l=None):
		vn_pred, _ = self.vn_crf.decode(pack, v_label, v_l)
		srl_pred, _ = self.srl_crf.decode(pack, v_label, v_l)
		return vn_pred, srl_pred


	def forward(self, pack):
		v_label, v_l = self.shared.v_label, self.shared.v_l

		if self.is_vn_active():
			vn_loss, vn_rs = self.vn_crf(pack)
			vn_pred, vn_gold = vn_rs['flat_pred'], vn_rs['flat_gold']
		else:
			vn_loss = 0

		if self.is_srl_active():
			srl_loss, srl_rs = self.srl_crf(pack)
			srl_pred, srl_gold = srl_rs['flat_pred'], srl_rs['flat_gold']
		else:
			srl_loss = 0

		if not self.shared.is_train:
			# a bit of analysis iff both crfs are active
			if self.is_vn_active() and self.is_srl_active():
				self.semlink_pred_vio_cnt += self.analyze_semlink(vn_pred, srl_pred)
				self.semlink_gold_vio_cnt += self.analyze_semlink(vn_gold, srl_gold) 

		loss = vn_loss * self.multi_w[0] + srl_loss * self.multi_w[1]
		rs = {}
		if self.is_vn_active():
			rs.update({'flat_vn_pred': vn_pred, 'flat_vn_gold': vn_gold})
		if self.is_srl_active():
			rs.update({'flat_srl_pred': srl_pred, 'flat_srl_gold': srl_gold})
		return loss, rs


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
							#print(self.opt.vn_class_map_inv[vnc_id], self.opt.roleset_suffix_map_inv[roleset_suffix_id])
							#print(mapping)
							#print(r_label, a_label)
							has_vio = True
							break
				if has_vio:
					vio_cnt += 1
				start += 1
		return vio_cnt


	# return a string of stats
	def print_cur_stats(self):
		vn_stats = self.vn_crf.print_cur_stats()
		srl_stats = self.srl_crf.print_cur_stats()
		return vn_stats

	# get training metric (scalar metric, extra metric)
	def get_epoch_metric(self):
		print(f'semlink pred violation: {self.semlink_pred_vio_cnt}/{self.vn_crf.num_prop}, {self.semlink_pred_vio_cnt/self.vn_crf.num_prop:.4f}')
		print(f'semlink gold violation: {self.semlink_gold_vio_cnt}/{self.vn_crf.num_prop}, {self.semlink_gold_vio_cnt/self.vn_crf.num_prop:.4f}')

		vn_f1, _ = self.vn_crf.get_epoch_metric()
		srl_f1, _ = self.srl_crf.get_epoch_metric()
		print('vn f1', vn_f1)
		print('srl f1', srl_f1)
		return (vn_f1 + srl_f1)/2, [vn_f1, srl_f1]


	def begin_pass(self):
		# clear stats
		self.vn_crf.begin_pass()
		self.srl_crf.begin_pass()
		self.semlink_gold_vio_cnt = 0
		self.semlink_pred_vio_cnt = 0

	def end_pass(self):
		self.vn_crf.end_pass()
		self.srl_crf.end_pass()

if __name__ == '__main__':
	pass



def _clean_error_file(errors_dict) -> str:
	errs = set(val for key, val in errors.dict.items())
	return os.linesep.join(errs)
