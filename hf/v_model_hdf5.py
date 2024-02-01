import sys
import argparse
import h5py
import numpy as np
import torch
from torch import nn
from torch import cuda
from util.holder import *
from util.util import *
from modules.pipeline import *
import traceback

#import spacy
#spacy_nlp = spacy.load('en')
# use nltk instead as it has better token-char mapping
import nltk
from nltk.tokenize import TreebankWordTokenizer
tb_tokenizer = TreebankWordTokenizer()


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--load_file', help="The path to pretrained model (optional)", default = "")
parser.add_argument('--dir', help="Path to the data dir", default="data/srl/")
parser.add_argument('--vn_label_dict', help="The path to VN label dictionary", default = "upc_vn.vn_label.dict")
parser.add_argument('--srl_label_dict', help="The path to SRL label dictionary", default = "upc_vn.srl_label.dict")
parser.add_argument('--roleset_dict', help="The path to roleset dictionary", default = "upc_vn.roleset_label.dict")
parser.add_argument('--roleset_suffix_dict', help="The path to roleset suffix (sense) dictionary", default = "upc_vn.roleset_suffix.dict")
## pipeline specs
parser.add_argument('--max_num_subtok', help="Maximal number subtokens in a word", type=int, default=8)
# bert specs
parser.add_argument('--bert_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")
parser.add_argument('--compact_mode', help="How word pieces be mapped to word level label", default='whole_word')
## pipeline stages
parser.add_argument('--enc', help="The type of encoder, e.g., bert", default='bert')
parser.add_argument('--cls', help="The type of classifier, e.g., linear", default='linear')
parser.add_argument('--loss', help="The type of losses, separated by ,; the first one MUST be role/crf", default='crf')
#
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)

def pad(ls, length, symbol, pad_back = True):
	if len(ls) >= length:
		return ls[:length]
	if pad_back:
		return ls + [symbol] * (length -len(ls))
	else:
		return [symbol] * (length -len(ls)) + ls

def process(opt, tokenizer, seq, predicates, v_idx):
	bos_tok, eos_tok = get_special_tokens(tokenizer)

	char_spans = list(tb_tokenizer.span_tokenize(seq))
	orig_toks = [seq[s:e] for s, e in char_spans]

	if v_idx is None:
		v_label = [next((i for i, span in enumerate(char_spans) if span == (seq.index(p), seq.index(p)+len(p))), None) for p in predicates if p in seq]
		v_label = [i for i in v_label if i is not None]
	else:
		v_label = v_idx

	if len(v_label) != len(predicates):
		print('valid predicates: ', ','.join([orig_toks[i] for i in v_label]))

	sent_subtoks = [tokenizer.tokenize(t) for t in orig_toks]
	tok_l = [len(subtoks) for subtoks in sent_subtoks]
	toks = [p for subtoks in sent_subtoks for p in subtoks]	# flatterning

	# pad for CLS and SEP
	CLS, SEP = tokenizer.cls_token, tokenizer.sep_token
	toks = [CLS] + toks + [SEP]
	tok_l = [1] + tok_l + [1]
	orig_toks = [CLS] + orig_toks + [SEP]
	v_label = [l+1 for l in v_label]

	tok_idx = np.array(tokenizer.convert_tokens_to_ids(toks), dtype=int)

	# note that the resulted actual seq length after subtoken collapsing could be different even within the same batch
	#	actual seq length is the origial sequence length
	#	seq length is the length after subword tokenization
	acc = 0
	sub2tok_idx = []
	for l in tok_l:
		sub2tok_idx.append(pad([p for p in range(acc, acc+l)], opt.max_num_subtok, -1))
		assert(len(sub2tok_idx[-1]) <= opt.max_num_subtok)
		acc += l
	sub2tok_idx = pad(sub2tok_idx, len(tok_idx), [-1 for _ in range(opt.max_num_subtok)])
	sub2tok_idx = np.array(sub2tok_idx, dtype=int)
	return tok_idx, sub2tok_idx, toks, orig_toks, v_label


def fix_opt(opt):
	opt.vn_label_dict = opt.dir + opt.vn_label_dict
	opt.srl_label_dict = opt.dir + opt.srl_label_dict
	opt.roleset_dict = opt.dir + opt.roleset_dict
	opt.roleset_suffix_dict = opt.dir + opt.roleset_suffix_dict
	opt.use_gold_predicate = 0
	opt.dropout = 0
	opt.lambd = ','.join([str(1) for _ in opt.loss.split(',')])
	return opt


def pretty_print_crf(opt, shared, m, pred_idx):
	batch_l = shared.batch_l
	orig_l = shared.orig_seq_l
	pred_log =[]
	for i in range(batch_l):
		orig_l_i = orig_l[i].item()	# convert to scalar
		a_pred_i = pred_idx[i, :orig_l_i, :orig_l_i]

		orig_tok_grouped = shared.res_map['orig_tok_grouped'][i][1:-1]
		pred_log.append(m.crf_loss.compose_log(orig_tok_grouped, a_pred_i[1:-1, 1:-1].cpu(), transpose=False))
	return pred_log


def pretty_print_v_class(opt, shared, v_labels, v_idx, v_score):
	batch_l = shared.batch_l
	orig_l = shared.orig_seq_l
	log = []
	for i in range(batch_l):
		orig_l_i = orig_l[i].item()	# convert to scalar
		label_idx = v_score[i, :orig_l_i, :].argmax(-1)
		log.append([f'{k}: {v_labels[label_idx[k]]}' for k in v_idx[i]])
	return log


def run(opt, shared, m, tokenizer, seq, predicates, v_idx, do_crf=True):
	tok_idx, sub2tok_idx, toks, orig_toks, v_idx = process(opt, tokenizer, seq, predicates, v_idx)
	print('predicate positions', v_idx)

	batch = Holder()
	batch.batch_l = 1
	batch.batch_ex_idx = [0]
	batch.tok_idx = to_device(Variable(torch.tensor([tok_idx]), requires_grad=False), opt.gpuid)
	batch.sub2tok_idx = to_device(torch.tensor([sub2tok_idx]).int(), opt.gpuid)
	batch.seq_l = len(tok_idx)
	batch.orig_seq_l = to_device(torch.tensor([len(orig_toks)]).int(), opt.gpuid)
	batch.semlink = None
	batch.semlink_l = None
	batch.srl_label = None
	batch.vn_label = None
	batch.roleset_ids = None
	batch.roleset_suffixes = None
	batch.v_label = to_device(torch.tensor([v_idx]).int(), opt.gpuid)
	batch.v_l = to_device(torch.tensor([len(v_idx)]).int(), opt.gpuid)
	batch.semlink_map = None
	batch.semlink_pool = None
	batch.res_map = {'orig_tok_grouped': [orig_toks]}

	m.update_context(batch)

	with torch.no_grad():
		pack = m.forward(batch, return_loss=False)

	log = pretty_print_v_class(opt, shared, opt.vn_labels, batch.v_label, pack['v_score'])

	return orig_toks[1:-1], log


def init(opt):
	opt = fix_opt(opt)
	opt = complete_opt(opt)
	opt.gpuid = -1
	opt.dropout = 0
	opt.lambd = ','.join([str(1) for _ in opt.loss.split(',')])
	shared = Holder()

	# load model
	m = Pipeline(opt, shared)
	print('loading pretrained model from {0}...'.format(opt.load_file))
	param_dict = load_param_dict('{0}.hdf5'.format(opt.load_file))
	m.set_param_dict(param_dict)
	m.distribute()

	tokenizer = AutoTokenizer.from_pretrained(opt.bert_type, add_special_tokens=False, use_fast=True)

	return opt, shared, m, tokenizer


def init_vnclass_model(dir_):
	opt = Holder()
	opt.dir = dir_
	opt.vn_label_dict = 'upc_vn.vn_label.dict'
	opt.srl_label_dict = 'upc_vn.srl_label.dict'
	opt.roleset_dict = 'upc_vn.roleset_label.dict'
	opt.roleset_suffix_dict = 'upc_vn.roleset_suffix.dict'
	opt.max_num_subtok = 8
	opt.bert_type = 'roberta-base'
	opt.compact_mode = 'whole_word'
	opt.enc = 'bert'
	opt.cls = 'vn_linear'
	opt.loss = 'vnclass'
	opt.gpuid = -1
	opt, shared, m, tokenizer = init(opt)
	


def main(args):
	opt = parser.parse_args(args)
	opt, shared, m, tokenizer = init(opt)

	seq = "He said he knows it."
	predicates = ['said', 'knows']
	orig_toks, log = run(opt, shared, m, tokenizer, seq, predicates, do_crf=False)

	print('###################################')
	print('Here is a sample prediction for input:')
	print('>> Input: ', seq)
	print(' '.join(orig_toks))
	print(log)

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

