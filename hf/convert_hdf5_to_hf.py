import sys
import argparse
from transformers import *
from util.holder import *
from util.util import *
from .roberta_for_srl import *
from modules import pipeline
from loss.crf_loss import CRFLoss
from loss.cross_crf_loss import CrossCRFLoss
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--load_file', help="Path to where HDF5 model to be loaded.", default="")
parser.add_argument('--dir', help="Path to the data dir", default="data/srl/")
parser.add_argument('--vn_label_dict', help="The path to VN label dictionary", default = "upc_vn.vn_label.dict")
parser.add_argument('--vn_class_dict', help="The path to VN class dictionary", default = "upc_vn.vn_class.dict")
parser.add_argument('--srl_label_dict', help="The path to SRL label dictionary", default = "upc_vn.srl_label.dict")
parser.add_argument('--roleset_dict', help="The path to roleset dictionary", default = "upc_vn.roleset_label.dict")
parser.add_argument('--roleset_suffix_dict', help="The path to roleset suffix (sense) dictionary", default = "upc_vn.roleset_suffix.dict")
# bert specs
parser.add_argument('--bert_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")
parser.add_argument('--compact_mode', help="How word pieces be mapped to word level label", default='whole_word')
## pipeline stages
parser.add_argument('--enc', help="The type of encoder, bert", default='bert')
parser.add_argument('--cls', help="The type of classifier, linear", default='linear')
parser.add_argument('--loss', help="The type of losses, separated by ,; the first one MUST be role/crf", default='crf')
#
parser.add_argument('--output', help="Path to output HuggingFace(HF) format", default='/models/hf')
parser.add_argument('--max_num_subtok', help="Maximal number subtokens in a word", type=int, default=8)


def main(args):
	opt = parser.parse_args(args)
	opt.vn_class_dict = opt.dir + opt.vn_class_dict
	opt.vn_label_dict = opt.dir + opt.vn_label_dict
	opt.srl_label_dict = opt.dir + opt.srl_label_dict
	opt.roleset_dict = opt.dir + opt.roleset_dict
	opt.roleset_suffix_dict = opt.dir + opt.roleset_suffix_dict
	opt = complete_opt(opt)
	opt.gpuid = -1
	opt.dropout = 0
	opt.lambd = ','.join([str(1) for _ in opt.loss.split(',')])
	shared = Holder()

	# load model
	m = pipeline.Pipeline(opt, shared)
	print('loading pretrained model from {0}...'.format(opt.load_file))
	param_dict = load_param_dict('{0}.hdf5'.format(opt.load_file))
	m.set_param_dict(param_dict)

	mlm = AutoModel.from_pretrained(opt.bert_type)
	tokenizer = AutoTokenizer.from_pretrained(opt.bert_type)

	config = mlm.config
	for k, v in opt.__dict__.items():
		setattr(config, k, v)
	config.gpuid = -1
	config.dropout = opt.dropout

	if 'roberta' in opt.bert_type:
		m_hf = RobertaForSRL(config, shared=Holder())
		m_hf.config = config
		# move parameters
		m_hf.roberta = m.encoder.bert
		m_hf.classifier = m.classifier
		if isinstance(m.loss[0], CRFLoss) or isinstance(m.loss[0], CrossCRFLoss):
			m_hf.crf_loss = m.loss[0]
		else:
			m_hf.v_class_loss = m.loss[0]
	else:
		raise Exception('unrecognized model type {0}'.format(opt.bert_type))
	
	m_hf.save_pretrained(opt.output)
	tokenizer.save_pretrained(opt.output)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))