import sys
import torch
from torch import nn
from torch import cuda
from torch.autograd import Variable
import numpy as np
import time
from .bert_encoder import *
from .bert_wp_encoder import *
from .bert_wpa_encoder import *
from .vn_linear_classifier import *
from .srl_linear_classifier import *
from .multi_classifier import *
from .merge_classifier import *
from .merge_concat_classifier import *
from .predicate_classifier import *
from .optimizer import *
from loss.crf_loss import *
from loss.multi_crf_loss import *
from loss.cross_crf_loss import *
from loss.unique_role_loss import *
from loss.semlink_loss import *
from loss.vnclass_loss import *
from loss.sense_loss import *
from util.holder import *
from util.util import *

class Pipeline(torch.nn.Module):
	def __init__(self, opt, shared):
		super(Pipeline, self).__init__()

		self.shared = shared
		self.opt = opt

		# pipeline stages
		if opt.enc == 'bert':
			self.encoder = BertEncoder(opt, shared)
		elif opt.enc == 'bert_wp':
			self.encoder = BertWPEncoder(opt, shared)
		elif opt.enc == 'bert_wpa':
			self.encoder = BertWPAEncoder(opt, shared)
		else:
			assert(False)

		if opt.cls == 'vn_linear':
			self.classifier = VNLinearClassifier(opt, shared)
		elif opt.cls == 'srl_linear':
			self.classifier = SRLLinearClassifier(opt, shared)
		elif opt.cls == 'joint':
			self.classifier = JointClassifier(opt, shared)
		elif opt.cls == 'predicate':
			self.classifier = PredicateClassifier(opt, shared)
		elif opt.cls == 'multi':
			self.classifier = MultiClassifier(opt, shared)
		elif opt.cls == 'merge':
			self.classifier = MergeClassifier(opt, shared)
		elif opt.cls == 'merge_concat':
			self.classifier = MergeConcatClassifier(opt, shared)
		else:
			assert(False)

		self.loss = []
		self.loss_names = self.opt.loss.split(',')
		for name in self.loss_names:
			if name == 'vn_crf' or name == 'srl_crf':
				self.loss.append(CRFLoss(name, opt, shared))
			elif name == 'cross_crf':
				self.loss.append(CrossCRFLoss(name, opt, shared))
			elif name == 'multi_crf':
				self.loss.append(MultiCRFLoss(name, opt, shared))
			elif name == 'predicate_crf':
				self.loss.append(PredicateCRFLoss(name, opt, shared))
			elif name == 'unique_role':
				self.loss.append(UniqueRoleLoss(name, opt, shared))
			elif name == 'semlink':
				self.loss.append(SemlinkLoss(name, opt, shared))
			elif name == 'vnclass':
				self.loss.append(VNClassLoss(name, opt, shared))
			elif name == 'sense':
				self.loss.append(SenseLoss(name, opt, shared))
			else:
				raise Exception("unrecognized loss {0}".format(name))

		self.loss = nn.ModuleList(self.loss)
		self.lambd = Variable(torch.Tensor([float(p) for p in opt.lambd.split(',')]), requires_grad=False)
		if self.opt.gpuid != -1:
			self.lambd = to_device(self.lambd, self.opt.gpuid)
		assert(len(self.lambd) == len(self.loss))


	def init_weight(self):
		missed_names = []
		if self.opt.param_init_type == 'xavier_uniform':
			for n, p in self.named_parameters():
				if p.requires_grad and not hasattr(p, 'skip_init'):
					if 'weight' in n:
						print('initializing {}'.format(n))
						nn.init.xavier_uniform_(p)
					elif 'bias' in n:
						print('initializing {}'.format(n))
						nn.init.constant_(p, 0)
					else:
						missed_names.append(n)
				else:
					missed_names.append(n)
		elif self.opt.param_init_type == 'xavier_normal':
			for n, p in self.named_parameters():
				if p.requires_grad and not hasattr(p, 'skip_init'):
					if 'weight' in n:
						print('initializing {}'.format(n))
						nn.init.xavier_normal_(p)
					elif 'bias' in n:
						print('initializing {}'.format(n))
						nn.init.constant_(p, 0)
					else:
						missed_names.append(n)
				else:
					missed_names.append(n)
		elif self.opt.param_init_type == 'no':
			for n, p in self.named_parameters():
				missed_names.append(n)
		else:
			assert(False)

		if len(missed_names) != 0:
			print('uninitialized fields: {0}'.format(missed_names)) 

		# Tie weights
		self.tie_weights()


	def tie_weights(self):
		if hasattr(self.encoder, 'tie_weights'):
			self.encoder.tie_weights()
		if hasattr(self.classifier, 'tie_weights'):
			self.classifier.tie_weights()


	def forward(self, batch, return_loss=True):
		# encoder
		enc = self.encoder(batch)

		# classifier
		pack = self.classifier(enc)

		if return_loss:
			batch_loss = to_device(torch.zeros(1), self.opt.gpuid)
			rs = {}
			for i, loss in enumerate(self.loss):
				if loss.is_active():
					l, _ = loss(pack)
				else:
					l = to_device(torch.zeros(1), self.opt.gpuid)
				batch_loss = batch_loss + l * self.lambd[i]
				rs[loss.name] = l
	
			pack.update(rs)
			pack['batch_loss'] = batch_loss
		return pack

	# update the contextual info of current batch
	def update_context(self, batch):
		self.shared.batch_ex_idx = batch.batch_ex_idx
		self.shared.batch_l = batch.batch_l
		self.shared.seq_l = batch.seq_l
		self.shared.orig_seq_l = batch.orig_seq_l
		self.shared.sub2tok_idx = batch.sub2tok_idx
		self.shared.semlink = batch.semlink
		self.shared.semlink_l = batch.semlink_l
		self.shared.srl_label = batch.srl_label
		self.shared.vn_label = batch.vn_label
		self.shared.vn_class = batch.vn_class
		self.shared.roleset_ids = batch.roleset_ids
		self.shared.roleset_suffixes = batch.roleset_suffixes
		self.shared.frame_idx = batch.frame_idx
		self.shared.frame_pool = batch.frame_pool
		self.shared.v_label = batch.v_label
		self.shared.v_l = batch.v_l
		self.shared.res_map = batch.res_map


	def begin_pass(self):
		self.encoder.begin_pass()
		self.classifier.begin_pass()
		for loss in self.loss:
			loss.begin_pass()

	def end_pass(self):
		self.encoder.end_pass()
		self.classifier.end_pass()
		for loss in self.loss:
			loss.end_pass()

	def print_cur_stats(self):
		log = []
		for l in self.loss:
			log.append(l.print_cur_stats())
		return ' '.join(log)

	def get_epoch_metric(self):
		head = None
		metrics = []
		for i, l in enumerate(self.loss):
			head_i, metrics_i = l.get_epoch_metric()
			# we only use the first loss metric to select model
			if i == 0:
				head = head_i
			metrics.extend(metrics_i)
		return head, metrics


	def distribute(self):
		if self.opt.gpuid == -1:
			return
		modules = []
		modules.append(self.encoder)
		modules.append(self.classifier)
		for loss in self.loss:
			modules.append(loss)

		for m in modules:
			# This is no longer needed
			#if hasattr(m, 'fp16') and  m.fp16:
			#	m.half()

			if hasattr(m, 'customize_cuda_id'):
				print('pushing module to customized cuda id: {0}'.format(m.customize_cuda_id))
				m.cuda(m.customize_cuda_id)
			else:
				print('pushing module to default cuda id: {0}'.format(self.opt.gpuid))
				m.cuda(self.opt.gpuid)


	def get_param_dict(self):
		is_cuda = self.opt.gpuid != -1
		param_dict = {}
		skipped_fields = []
		for n, p in self.named_parameters():
			# save all parameters that do not have skip_save flag
			# 	unlearnable parameters will also be saved
			if not hasattr(p, 'skip_save') or p.skip_save == 0:
				param_dict[n] =  torch2np(p.data, is_cuda)
			else:
				skipped_fields.append(n)
		#print('skipped fields:', skipped_fields)
		return param_dict


	def set_param_dict(self, param_dict, verbose=True):
		skipped_fields = []
		rec_fields = []
		for n, p in self.named_parameters():
			if n in param_dict:
				rec_fields.append(n)
				# load everything we have
				if verbose:
					print('setting {0}'.format(n))
				p.data.copy_(torch.from_numpy(param_dict[n][:]))
			else:
				skipped_fields.append(n)
		print('skipped fileds: {0}'.format(skipped_fields))
