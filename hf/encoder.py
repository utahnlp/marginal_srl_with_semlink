import sys
import torch
from torch import cuda
from transformers import *
from torch import nn

# encoder for inputs with predicate (WP)
class WPEncoder(torch.nn.Module):
	def __init__(self, opt, shared):
		super(WPEncoder, self).__init__()
		self.opt = opt
		self.shared = shared

		self.bert = AutoModel.from_pretrained(self.opt.bert_type)
		tokenizer = AutoTokenizer.from_pretrained(self.opt.bert_type)
		self.pad_token_id = tokenizer.pad_token_id


	def forward(self, batch):
		tok_idx = batch.tok_idx
		wp_idx = batch.wp_idx
		assert(self.shared.seq_l == tok_idx.shape[1])

		max_wp_l = (wp_idx != self.pad_token_id).sum(-1).max().long()
		max_concat_l = tok_idx.shape[1] + max_wp_l
		concated_input = torch.cat([tok_idx, wp_idx], dim=1)		
		concated_input = concated_input[:, :max_concat_l]

		last, pooled = self.bert(concated_input, return_dict=False)
		last = last + pooled.unsqueeze(1) * 0.0

		# only take the tok_idx output
		last = last[:, :tok_idx.shape[1]]
		
		return last


