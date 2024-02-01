import sys
import argparse
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch import nn
from torch import cuda
from util.holder import *
from util.data import *
from modules.pipeline import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', help="Path to the data dir", default="data/srl/")
parser.add_argument('--data', help="Path to test data hdf5 file.", default="upc_vn.test.hdf5")
parser.add_argument('--load_file', help="The path to pretrained model (optional)", default="")
parser.add_argument('--vn_label_dict', help="The path to VN label dictionary", default = "upc_vn.vn_label.dict")
parser.add_argument('--vn_class_dict', help="The path to VN class dictionary", default = "upc_vn.vn_class.dict")
parser.add_argument('--srl_label_dict', help="The path to SRL label dictionary", default = "upc_vn.srl_label.dict")
parser.add_argument('--roleset_dict', help="The path to roleset dictionary", default = "upc_vn.roleset_label.dict")
parser.add_argument('--roleset_suffix_dict', help="The path to roleset suffix (sense) dictionary", default = "upc_vn.roleset_suffix.dict")
parser.add_argument('--arg_role_map', help="The path to valid arg-role mapping for semlink", default = "upc_vn_arg_role.txt")
# resource specs
parser.add_argument('--res', help="Path to test resource files, seperated by comma.", default="")
## pipeline specs
parser.add_argument('--max_num_subtok', help="Maximal number subtokens in a word", type=int, default=8)
parser.add_argument('--dropout', help="The dropout probability", type=float, default=0.0)
parser.add_argument('--param_init_type', help="The type of parameter initialization", default='xavier_normal')
# bert specs
parser.add_argument('--bert_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")
parser.add_argument('--compact_mode', help="How word pieces be mapped to word level label", default='whole_word')
## pipeline stages
parser.add_argument('--enc', help="The type of encoder, bert", default='bert')
parser.add_argument('--cls', help="The type of classifier, linear", default='linear')
parser.add_argument('--loss', help="The type of losses, separated by ,; the first one MUST be role/crf", default='crf')
parser.add_argument('--lambd', help="The weight of losses, separated by ,; ignored if only one loss", default='1.0')
parser.add_argument('--multi_crf_lambd', help="The weight of multi_crf losses, separated by ,; ignored if only one loss", default='1,1')
parser.add_argument('--data_flag', help="Flag for data batches", default="data")
#
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--seed', help="The random seed", type=int, default=3435)
parser.add_argument('--verbose', help="Whether to print out every prediction", type=int, default=0)
parser.add_argument('--neg_inf', help="The approx. negative infinity used in eval.", type=float, default=-1e3)
#
parser.add_argument('--use_gold_predicate', help="Whether to hard code gold vn classes during evaluation", type=int, default=1)
parser.add_argument('--conll_output', help="The prefix of conll formated output", default='upc_vn')
parser.add_argument('--logs', help="The list of logs, separated by comma", default='')
#
parser.add_argument('--eval_with_semlink', help="Whether to apply semlink constraint during evaluation.", type=int, default=0)
parser.add_argument('--eval_with_frame', help="Whether to apply frame core role constraint during evaluation.", type=int, default=0)
parser.add_argument('--eval_with_srl', help="Whether to apply gold srl labels to decoding.", type=int, default=0)
#
parser.add_argument('--sent_filter', help="Only eval for sentences in this file (tokenized).", default="")
#
parser.add_argument('--multi_w', help="The weight of multi_crf losses, separated by ,; ignored if only one loss", default='1,1')
parser.add_argument('--marginal_w', help="The lambda for marginal CRF loss", type=float, default=1)


def evaluate(opt, shared, m, data):
	m.train(False)
	shared.is_train = False

	val_loss = 0.0
	num_ex = 0
	num_batch = 0

	val_idx, val_num_ex = data.subsample(1.0)
	data_size = val_idx.size()[0]
	print('validating on the {0} batches {1} examples...'.format(data_size, val_num_ex))

	m.begin_pass()
	data.begin_pass(val_idx)
	for i in tqdm(range(data_size), desc="predicting"):
		batch = data[i]
		batch_l = batch.batch_l

		shared.batch_flag = opt.data_flag

		with torch.no_grad():
			m.update_context(batch)
			pack = m.forward(batch)
			batch_loss = pack['batch_loss']

		# stats
		val_loss += float(batch_loss.item())
		num_ex += batch_l
		num_batch += 1

	print(m.print_cur_stats())
	perf, extra_perf = m.get_epoch_metric()
	data.end_pass()
	m.end_pass()
	return (perf, extra_perf, val_loss / num_batch, num_ex)



def main(args):
	opt = parser.parse_args(args)
	shared = Holder()

	# 
	opt.data = opt.dir + opt.data
	opt.res = '' if opt.res == ''  else ','.join([opt.dir + path for path in opt.res.split(',')])
	opt.vn_label_dict = opt.dir + opt.vn_label_dict
	opt.vn_class_dict = opt.dir + opt.vn_class_dict
	opt.srl_label_dict = opt.dir + opt.srl_label_dict
	opt.roleset_dict = opt.dir + opt.roleset_dict
	opt.roleset_suffix_dict = opt.dir + opt.roleset_suffix_dict
	opt.sent_filter = opt.dir + opt.sent_filter
	opt.arg_role_map = opt.dir + opt.arg_role_map
	opt = complete_opt(opt)

	torch.manual_seed(opt.seed)
	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
		torch.cuda.manual_seed_all(opt.seed)

	# build model
	m = Pipeline(opt, shared)

	# initializing from pretrained
	m.init_weight()
	print('loading pretrained model from {0}...'.format(opt.load_file))
	param_dict = load_param_dict('{0}.hdf5'.format(opt.load_file))
	m.set_param_dict(param_dict)

	if opt.gpuid != -1:
		m.distribute()	# distribute to multigpu

	# loading data
	res_files = None if opt.res == '' else opt.res.split(',')
	data = Data(opt, opt.data, res_files)

	#
	perf, extra_perf, avg_loss, num_ex = evaluate(opt, shared, m, data)
	extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_perf])
	print('Val {0:.4f} Extra {1} Loss: {2:.4f}'.format(
		perf, extra_perf_str, avg_loss))

	#print('saving model to {0}'.format('tmp'))
	#param_dict = m.get_param_dict()
	#save_param_dict(param_dict, '{0}.hdf5'.format('tmp'))


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))