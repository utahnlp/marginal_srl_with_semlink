import sys
import argparse
import h5py
import numpy as np
import torch
from torch import nn
from torch import cuda
from util.holder import *
from util.util import *
from hf.roberta_for_srl import *
import traceback

#import spacy
#spacy_nlp = spacy.load('en')
# use nltk instead as it has better token-char mapping
import nltk
from nltk.tokenize import TreebankWordTokenizer
tb_tokenizer = TreebankWordTokenizer()

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--load_file', help="The path to pretrained model (optional)", default = "")


def pad(ls, length, symbol, pad_back = True):
	if len(ls) >= length:
		return ls[:length]
	if pad_back:
		return ls + [symbol] * (length -len(ls))
	else:
		return [symbol] * (length -len(ls)) + ls


class VModel:
	def __init__(self, model_path):
		self.opt = Holder()
		self.opt.use_gold_predicate = 0
		self.opt.dropout = 0
		self.shared = Holder()

		self.m = RobertaForSRL.from_pretrained(model_path, global_opt=self.opt, shared=self.shared)
		self.opt.__dict__.update(self.m.config.__dict__)
		self.tokenizer = AutoTokenizer.from_pretrained(self.opt.bert_type, add_special_tokens=False, use_fast=True)
		
		if self.opt.gpuid != -1:
			self.m.cuda(self.opt.gpuid)

	def process(self, seq, predicates, v_idx):
		bos_tok, eos_tok = get_special_tokens(self.tokenizer)
	
		char_spans = list(tb_tokenizer.span_tokenize(seq))
		orig_toks = [seq[s:e] for s, e in char_spans]
	
		if v_idx is None:
			v_label = [next((i for i, span in enumerate(char_spans) if span == (seq.index(p), seq.index(p)+len(p))), None) for p in predicates 	if p in seq]
			v_label = [i for i in v_label if i is not None]
		else:
			v_label = v_idx
	
		if len(v_label) != len(predicates):
			print('valid predicates: ', ','.join([orig_toks[i] for i in v_label]))
	
		sent_subtoks = [self.tokenizer.tokenize(t) for t in orig_toks]
		tok_l = [len(subtoks) for subtoks in sent_subtoks]
		toks = [p for subtoks in sent_subtoks for p in subtoks]	# flatterning
	
		# pad for CLS and SEP
		CLS, SEP = self.tokenizer.cls_token, self.tokenizer.sep_token
		toks = [CLS] + toks + [SEP]
		tok_l = [1] + tok_l + [1]
		orig_toks = [CLS] + orig_toks + [SEP]
		v_label = [l+1 for l in v_label]
	
		tok_idx = np.array(self.tokenizer.convert_tokens_to_ids(toks), dtype=int)
	
		# note that the resulted actual seq length after subtoken collapsing could be different even within the same batch
		#	actual seq length is the origial sequence length
		#	seq length is the length after subword tokenization
		acc = 0
		sub2tok_idx = []
		for l in tok_l:
			sub2tok_idx.append(pad([p for p in range(acc, acc+l)], self.opt.max_num_subtok, -1))
			assert(len(sub2tok_idx[-1]) <= self.opt.max_num_subtok)
			acc += l
		sub2tok_idx = pad(sub2tok_idx, len(tok_idx), [-1 for _ in range(self.opt.max_num_subtok)])
		sub2tok_idx = np.array(sub2tok_idx, dtype=int)
		return tok_idx, sub2tok_idx, toks, orig_toks, v_label
	

	def get_v_class(self, v_labels, v_idx, v_score):
		batch_l = self.shared.batch_l
		orig_l = self.shared.orig_seq_l
		v_class = []
		for i in range(batch_l):
			orig_l_i = orig_l[i].item()	# convert to scalar
			label_idx = v_score[i, :orig_l_i, :].argmax(-1)
			v_class.append([v_labels[label_idx[k]] for k in v_idx[i]])
		return v_class


	def run_raw(self, seq, predicates, v_idx, v_type):
		tok_idx, sub2tok_idx, toks, orig_toks, v_idx = self.process(seq, predicates, v_idx)

		self.m.update_context(orig_seq_l=to_device(torch.tensor([len(orig_toks)]).int(), self.opt.gpuid), 
			sub2tok_idx=to_device(torch.tensor([sub2tok_idx]).int(), self.opt.gpuid), 
			res_map={'orig_tok_grouped': [orig_toks]})

		tok_idx = to_device(Variable(torch.tensor([tok_idx]), requires_grad=False), self.opt.gpuid)
		v_l = to_device(torch.Tensor([len(v_idx)]).long().view(1), self.opt.gpuid)
		v_idx = to_device(torch.Tensor(v_idx).long().view(1, -1), self.opt.gpuid)

		with torch.no_grad():
			pack = self.m.forward(tok_idx, v_idx, v_l, do_crf=False)

		if v_type == 'vn':
			v_labels = self.opt.vn_classes
			v_score = pack['v_score']
			# block out non-v classes
			#v_mask = torch.tensor(get_v_mask(v_labels)).float()
			#v_score = v_score + (1.0 - v_mask.view(1, -1)) * -1e4
		elif v_type == 'srl':
			v_labels = self.opt.roleset_suffix
			v_score = pack['sense_score']
			# block out the 'O'
			v_mask = torch.ones(len(v_labels))
			v_mask[0] = 0.0
			v_score = v_score + (1.0 - v_mask.view(1, -1)) * -1e4
		else:
			raise Exception('unrecognized v_type', v_type)

		v_class = self.get_v_class(v_labels, v_idx, v_score)
		return orig_toks, v_class


	def run_preprocessed(self, tok_idx, sub2tok_idx, toks, orig_toks, v_idx, v_type):
		self.m.update_context(orig_seq_l=to_device(torch.tensor([len(orig_toks)]).int(), self.opt.gpuid), 
			sub2tok_idx=to_device(torch.tensor([sub2tok_idx]).int(), self.opt.gpuid), 
			res_map={'orig_tok_grouped': [orig_toks]})

		tok_idx = to_device(Variable(torch.tensor([tok_idx]), requires_grad=False), self.opt.gpuid)
		v_l = to_device(torch.Tensor([len(v_idx)]).long().view(1), self.opt.gpuid)
		v_idx = to_device(torch.Tensor(v_idx).long().view(1, -1), self.opt.gpuid)

		with torch.no_grad():
			pack = self.m.forward(tok_idx, v_idx, v_l, do_crf=False)

		if v_type == 'vn':
			v_labels = self.opt.vn_classes
			v_score = pack['v_score']
		elif v_type == 'srl':
			v_labels = self.opt.roleset_suffix
			v_score = pack['sense_score']
		else:
			raise Exception('unrecognized v_type', v_type)

		v_class = self.get_v_class(v_labels, v_idx, v_score)
		return v_class


def main(args):
	opt = parser.parse_args(args)
	model = VModel(opt.load_file)

	#seq = "He said he knows it."
	#predicates = ['said', 'knows']
	#v_idx = [1, 3]
	seq = "The police did a drive around with the cabby."
	predicates = ['drive']
	#predicates = []
	orig_toks, v_class = model.run_raw(seq, predicates, v_idx=None, v_type='vn')

	print('###################################')
	print('Here is a sample prediction for input:')
	print('>> Input: ', seq)
	print(' '.join(orig_toks))
	print(v_class)

	#print('###################################')
	#print('#           Instructions          #')
	#print('###################################')
	#print('>> Enter a input senquence as prompted.')

	#while True:
	#	try:
	#		print('###################################')
	#		seq = input("Enter a sequence: ")
	#		orig_toks, log = run(opt, shared, m, tokenizer, seq, predicates, do_crf=False)
	#		print(' '.join(orig_toks))
	#		print(log)
#
	#	except KeyboardInterrupt:
	#		return
	#	except BaseException as e:
	#		traceback.print_tb(e.__traceback__)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

