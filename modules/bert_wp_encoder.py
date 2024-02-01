import sys
import torch
from torch import cuda
from transformers import *
from torch import nn
from torch.autograd import Variable
from util.holder import *
from util.util import *

# encoder for inputs with predicate (WP)
class BertWPEncoder(torch.nn.Module):
	def __init__(self, opt, shared):
		super(BertWPEncoder, self).__init__()
		self.opt = opt
		self.shared = shared

		self.zero = Variable(torch.zeros(1), requires_grad=False)
		self.zero = to_device(self.zero, self.opt.gpuid)
		
		print('loading transformer...')
		self.bert = AutoModel.from_pretrained(self.opt.bert_type)
		tokenizer = AutoTokenizer.from_pretrained(self.opt.bert_type)
		self.pad_token_id = tokenizer.pad_token_id

		for n in self.bert.children():
			for p in n.parameters():
				p.skip_init = True


	def forward(self, batch):
		tok_idx = to_device(batch.tok_idx, self.opt.gpuid)
		wp_idx = to_device(batch.wp_idx, self.opt.gpuid)
		assert(self.shared.seq_l == tok_idx.shape[1])

		max_wp_l = (wp_idx != self.pad_token_id).sum(-1).max().long()
		max_concat_l = tok_idx.shape[1] + max_wp_l
		concated_input = torch.cat([tok_idx, wp_idx], dim=1)		
		concated_input = concated_input[:, :max_concat_l]

		last, pooled = self.bert(concated_input, return_dict=False)
		last = last + pooled.unsqueeze(1) * self.zero

		# only take the tok_idx output
		last = last[:, :tok_idx.shape[1]]

		# move to the original device
		last = to_device(last, self.opt.gpuid)
		
		return last


	def begin_pass(self):
		pass

	def end_pass(self):
		pass


