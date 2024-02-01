import sys
import torch
from torch import cuda
from transformers import *
from torch import nn
from torch.autograd import Variable
from util.holder import *
from util.util import *

class BertEncoder(torch.nn.Module):
	def __init__(self, opt, shared):
		super(BertEncoder, self).__init__()
		self.opt = opt
		self.shared = shared

		self.zero = Variable(torch.zeros(1), requires_grad=False)
		self.zero = to_device(self.zero, self.opt.gpuid)
		
		print('loading transformer...')
		self.bert = AutoModel.from_pretrained(self.opt.bert_type)

		for n in self.bert.children():
			for p in n.parameters():
				p.skip_init = True


	def forward(self, batch):
		tok_idx = to_device(batch.tok_idx, self.opt.gpuid)

		last, pooled = self.bert(tok_idx, return_dict=False)

		last = last + pooled.unsqueeze(1) * self.zero

		# move to the original device
		last = to_device(last, self.opt.gpuid)
		
		return last


	def begin_pass(self):
		pass

	def end_pass(self):
		pass


