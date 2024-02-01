import sys
import argparse
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch import cuda
from util.holder import *
from util.util import *
from util.data import *
from modules.optimizer import *
from modules.pipeline import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', help="Path to the data dir", default="data/srl/")
parser.add_argument('--train_data', help="Path to training data hdf5 file.", default="upc_vn.train.hdf5")
parser.add_argument('--val_data', help="Path to validation data hdf5 file.", default="upc_vn.val.hdf5")
parser.add_argument('--extra_data', help="Path to extra training data hdf5 file.", default="")
parser.add_argument('--save_file', help="Path to where model to be saved.", default="model")
parser.add_argument('--load_file', help="The path to pretrained model (optional)", default = "")
parser.add_argument('--vn_label_dict', help="The path to VN label dictionary", default = "upc_vn.vn_label.dict")
parser.add_argument('--vn_class_dict', help="The path to VN class dictionary", default = "upc_vn.vn_class.dict")
parser.add_argument('--srl_label_dict', help="The path to SRL label dictionary", default = "upc_vn.srl_label.dict")
parser.add_argument('--roleset_dict', help="The path to roleset dictionary", default = "upc_vn.roleset_label.dict")
parser.add_argument('--roleset_suffix_dict', help="The path to roleset suffix (sense) dictionary", default = "upc_vn.roleset_suffix.dict")
parser.add_argument('--arg_role_map', help="The path to valid arg-role mapping for semlink", default = "upc_vn_arg_role.txt")
# resource specs
parser.add_argument('--train_res', help="Path to training resource files, seperated by comma.", default="")
parser.add_argument('--val_res', help="Path to validation resource files, seperated by comma.", default="")
parser.add_argument('--extra_res', help="Path to extra training resource files, separated by comma.", default="")
parser.add_argument('--data_flag', help="Flag for data batches", default="data")
parser.add_argument('--extra_flag', help="Flag for extra data batches", default="extra")
## pipeline specs
parser.add_argument('--max_num_subtok', help="Maximal number subtokens in a word", type=int, default=8)
parser.add_argument('--hidden_size', help="The general hidden size of the pipeline", type=int, default=768)
parser.add_argument('--dropout', help="The dropout probability", type=float, default=0.1)
parser.add_argument('--percent', help="The percent of training data to use", type=float, default=1.0)
parser.add_argument('--val_percent', help="The percent of validation data to use", type=float, default=1.0)
parser.add_argument('--extra_percent', help="The percent of extra data to use", type=float, default=1.0)
parser.add_argument('--epochs', help="The number of epoches for training", type=int, default=3)
parser.add_argument('--optim', help="The name of optimizer to use for training", default='adamw_fp16')
parser.add_argument('--learning_rate', help="The learning rate for training", type=float, default=0.001)
parser.add_argument('--clip', help="The norm2 threshold to clip, set it to negative to disable", type=float, default=1.0)
parser.add_argument('--adam_betas', help="The betas used in adam", default='0.9,0.999')
parser.add_argument('--weight_decay', help="The factor of weight decay", type=float, default=0.01)
# bert specs
parser.add_argument('--bert_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")
parser.add_argument('--warmup_perc', help="The percentages of total expectec updates to warmup", type=float, default=0.1)
parser.add_argument('--compact_mode', help="How word pieces be mapped to word level label", default='whole_word')
## pipeline stages
parser.add_argument('--enc', help="The type of encoder, bert", default='bert')
parser.add_argument('--cls', help="The type of classifier, linear", default='linear')
parser.add_argument('--loss', help="The type of losses, separated by ,; the first one MUST be role/crf", default='crf')
parser.add_argument('--lambd', help="The weight of losses, separated by ,; ignored if only one loss", default='1.0')
#
parser.add_argument('--param_init_type', help="The type of parameter initialization", default='xavier_normal')
parser.add_argument('--print_every', help="Print stats after this many batches", type=int, default=200)
parser.add_argument('--seed', help="The random seed", type=int, default=1)
parser.add_argument('--shuffle_seed', help="The random seed specifically for shuffling", type=int, default=1)
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--acc_batch_size', help="The accumulative batch size, -1 to disable", type=int, default=-1)
parser.add_argument('--neg_inf', help="The approx. negative infinity used in train.", type=float, default=-1e3)
#
parser.add_argument('--use_gold_predicate', help="Whether to hard code vn classes during evaluation", type=int, default=1)
parser.add_argument('--conll_output', help="The prefix of conll formated output", default='upc_vn')
#
parser.add_argument('--lambda_v', help="The weight on predicate classifier loss", type=float, default=0)
#
parser.add_argument('--eval_with_semlink', help="Whether to apply semlink constraint during evaluation.", type=int, default=0)
parser.add_argument('--eval_with_frame', help="Whether to apply frame core role constraint during evaluation.", type=int, default=0)
parser.add_argument('--eval_with_srl', help="Whether to apply gold srl labels to decoding.", type=int, default=0)
# CRF related
parser.add_argument('--multi_w', help="The weight of multi_crf losses, separated by ,; ignored if only one loss", default='1,1')
parser.add_argument('--marginal_w', help="The lambda for marginal CRF loss", type=float, default=1)
parser.add_argument('--srl_unique_sample', help="Number of samples in SRL unique core marginal crf", type=int, default=1)


# train batch by batch, accumulate batches until the size reaches acc_batch_size
def train_epoch(opt, shared, m, optim, epoch_id, data, sub_idx, extra_data=None, extra_idx=None):
	train_loss = 0.0
	num_ex = 0
	num_batch = 0
	start_time = time.time()
	min_grad_norm2 = 1000000000000.0
	max_grad_norm2 = 0.0

	all_loss = {loss_name: 0.0 for loss_name in m.loss_names}

	# subsamples of data
	# if subsample indices provided, permutate from subsamples
	#	else permutate from all the data
	data_size = sub_idx.size()[0]
	batch_order = torch.randperm(data_size)
	batch_order = sub_idx[batch_order]
	all_data = []
	for i in range(data_size):
		all_data.append((data, batch_order[i]))

	if extra_data is not None:
		extra_size = extra_idx.size()[0]
		extra_batch_order = torch.randperm(extra_size)
		extra_batch_order = extra_idx[extra_batch_order]
		all_extra_data = []
		for i in range(extra_size):
			all_extra_data.append((extra_data, extra_batch_order[i]))
		# shuffle the combined data, in an interleaved way
		all_data = random_interleave(all_data, all_extra_data)
		data_size = len(all_data)
		print('combined with extra train data, num batches: {0}'.format(data_size))

	acc_batch_size = 0
	shared.is_train = True
	m.train(True)
	m.begin_pass()
	data.begin_pass(batch_order)
	if extra_data:
		extra_data.begin_pass(extra_batch_order)
	for i in range(data_size):
		shared.epoch = epoch_id
		shared.has_gold = True
		shared.data_size = data_size

		cur_data, cur_idx = all_data[i]
		batch = cur_data[cur_idx]
		batch_l = batch.batch_l

		if cur_data == data:
			shared.batch_flag = opt.data_flag
		else:
			shared.batch_flag = opt.extra_flag

		# fwd pass
		m.update_context(batch)
		pack = m.forward(batch)
		batch_loss = pack['batch_loss']

		# stats
		train_loss += float(batch_loss.item())
		for loss_name in m.loss_names:
			all_loss[loss_name] += pack[loss_name].item()
		num_ex += batch_l
		num_batch += 1
		time_taken = time.time() - start_time
		acc_batch_size += batch_l

		# accumulate grads
		grad_norm2 = optim.backward(m, batch_loss)

		# accumulate current batch until the rolled up batch size exceeds threshold or meet certain boundary
		if i == data_size-1 or acc_batch_size >= opt.acc_batch_size or (i+1) % opt.print_every == 0:
			optim.step(m)
			shared.num_update += 1

			# clear up grad
			m.zero_grad()
			acc_batch_size = 0

			# stats
			grad_norm2_avg = grad_norm2
			min_grad_norm2 = min(min_grad_norm2, grad_norm2_avg)
			max_grad_norm2 = max(max_grad_norm2, grad_norm2_avg)
			time_taken = time.time() - start_time

			if (i+1) % opt.print_every == 0:
				stats = '{0}, Batch {1:.1f}k '.format(epoch_id+1, float(i+1)/1000)
				stats += 'Grad {0:.1f}/{1:.1f} '.format(min_grad_norm2, max_grad_norm2)
				stats += 'Loss {0:.4f} '.format(train_loss / num_batch)	# just a rough estimate
				for loss_name in m.loss_names:
					stats += '{0}: {1:.4f} '.format(loss_name, float(all_loss[loss_name]) / num_batch)
				stats += m.print_cur_stats()
				stats += ' Time {0:.1f}'.format(time_taken)
				print(stats)

	print(m.print_cur_stats())
	perf, extra_perf = m.get_epoch_metric()
	data.end_pass()
	if extra_data:
		extra_data.end_pass()
	m.end_pass()

	return perf, extra_perf, train_loss / num_batch

def train(opt, shared, m, optim, train_data, val_data, extra_data):
	best_val_perf = -1.0	# something < 0
	test_perf = 0.0
	train_perfs = []
	val_perfs = []
	extra_perfs = []

	# if the specified shuffle seed is not the same as default seed,
	#	force it a random shuffle
	if opt.shuffle_seed != opt.seed:
		torch.manual_seed(opt.seed)

	train_idx, train_num_ex = train_data.subsample(opt.percent)
	print('{0} examples sampled for training'.format(train_num_ex))
	print('for the record, the first 10 training batches are: {0}'.format(train_idx[:10]))
	# sample the same proportion from the dev set as well
	#	but we don't want this to be too small
	minimal_dev_num = max(int(train_num_ex * 0.1), 1000)
	val_idx, val_num_ex = val_data.subsample(opt.val_percent, minimal_num=minimal_dev_num)
	print('{0} examples sampled for dev'.format(val_num_ex))
	print('for the record, the first 10 dev batches are: {0}'.format(val_idx[:10]))

	if extra_data:
		extra_idx, extra_num_ex = extra_data.subsample(opt.extra_percent)
		print('{0} examples sampled for extra'.format(extra_num_ex))
		print('for the record, the first 10 extra batches are: {0}'.format(extra_idx[:10]))
	else:
		extra_idx = None
		extra_num_ex = 0

	shared.num_train_ex = train_num_ex + extra_num_ex
	shared.num_update = 0
	start = 0
	for i in range(start, opt.epochs):
		train_perf, extra_train_perf, loss = train_epoch(opt, shared, m, optim, i, train_data, train_idx, extra_data, extra_idx)
		train_perfs.append(train_perf)
		extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_train_perf])
		print('Train {0:.4f} All {1}'.format(train_perf, extra_perf_str))

		# evaluate
		#	and save if it's the best model
		val_perf, extra_val_perf, val_loss = validate(opt, shared, m, val_data, val_idx)
		val_perfs.append(val_perf)
		extra_perfs.append(extra_val_perf)
		extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_val_perf])
		print('Val {0:.4f} All {1}'.format(val_perf, extra_perf_str))

		perf_table_str = ''
		cnt = 0
		print('Epoch  | Train | Val ...')
		for train_perf, extra_perf in zip(train_perfs, extra_perfs):
			extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_perf])
			perf_table_str += '{0}\t{1:.4f}\t{2}\n'.format(cnt+1, train_perf, extra_perf_str)
			cnt += 1
		print(perf_table_str)

		if val_perf > best_val_perf:
			best_val_perf = val_perf
			print('saving model to {0}'.format(opt.save_file))
			param_dict = m.get_param_dict()
			save_param_dict(param_dict, '{0}.hdf5'.format(opt.save_file))
			save_opt(opt, '{0}.opt'.format(opt.save_file))

		else:
			print('skip saving model for perf <= {0:.4f}'.format(best_val_perf))



def validate(opt, shared, m, val_data, val_idx):
	m.train(False)
	shared.is_train = False

	val_loss = 0.0
	num_ex = 0
	num_batch = 0

	data_size = val_idx.size()[0]
	all_val = []
	for i in range(data_size):
		all_val.append((val_data, val_idx[i]))

	#data_size = val_idx.size()[0]
	print('validating on the {0} batches...'.format(data_size))

	m.begin_pass()
	val_data.begin_pass(val_idx)
	for i in range(data_size):
		cur_data, cur_idx = all_val[i]
		batch = cur_data[cur_idx]
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

	perf, extra_perf = m.get_epoch_metric()	# we only use the first loss's corresponding metric to select models
	val_data.end_pass()
	m.end_pass()
	return (perf, extra_perf, val_loss / num_batch)


def main(args):
	opt = parser.parse_args(args)
	shared = Holder()

	# 
	opt.train_data = opt.dir + opt.train_data
	opt.val_data = opt.dir + opt.val_data
	opt.extra_data = opt.dir + opt.extra_data
	opt.train_res = '' if opt.train_res == ''  else ','.join([opt.dir + path for path in opt.train_res.split(',')])
	opt.val_res = '' if opt.val_res == ''  else ','.join([opt.dir + path for path in opt.val_res.split(',')])
	opt.extra_res = '' if opt.extra_res == ''  else ','.join([opt.dir + path for path in opt.extra_res.split(',')])
	opt.vn_label_dict = opt.dir + opt.vn_label_dict
	opt.vn_class_dict = opt.dir + opt.vn_class_dict
	opt.srl_label_dict = opt.dir + opt.srl_label_dict
	opt.roleset_dict = opt.dir + opt.roleset_dict
	opt.roleset_suffix_dict = opt.dir + opt.roleset_suffix_dict
	opt.arg_role_map = opt.dir + opt.arg_role_map

	opt = complete_opt(opt)

	torch.manual_seed(opt.seed)
	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
		torch.cuda.manual_seed_all(opt.seed)

	print(opt)

	# build model
	m = Pipeline(opt, shared)
	optim = get_optimizer(opt, shared)

	# initializing from pretrained
	if opt.load_file != '':
		m.init_weight()
		print('loading pretrained model from {0}...'.format(opt.load_file))
		param_dict = load_param_dict('{0}.hdf5'.format(opt.load_file))
		m.set_param_dict(param_dict)
	else:
		m.init_weight()
	
	model_parameters = filter(lambda p: p.requires_grad, m.parameters())
	num_params = sum([np.prod(p.size()) for p in model_parameters])	
	print('total number of trainable parameters: {0}'.format(num_params))
	
	if opt.gpuid != -1:
		m.distribute()	# distribute to multigpu
	m = optim.build_optimizer(m)	# build optimizer after distributing model to devices

	# loading data
	train_res_files = None if opt.train_res == '' else opt.train_res.split(',')
	train_data = Data(opt, opt.train_data, train_res_files)
	val_res_files = None if opt.val_res == '' else opt.val_res.split(',')
	val_data = Data(opt, opt.val_data, val_res_files)
	extra_res_files = None if opt.extra_res == '' else opt.extra_res.split(',')
	extra_data = None if not extra_res_files else Data(opt, opt.extra_data, extra_res_files)

	print('{0} batches in train set'.format(train_data.size()))
	if extra_data:
		print('{0} batches in extra set'.format(extra_data.size()))

	train(opt, shared, m, optim, train_data, val_data, extra_data)



if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))