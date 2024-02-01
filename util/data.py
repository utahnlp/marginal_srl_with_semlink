import io
import h5py
import torch
from torch import nn
from torch import cuda
import numpy as np
import ujson
from .util import *
from .holder import *
from tqdm import tqdm

class Data():
	def __init__(self, opt, data_file, res_files=None, triple_mode=False, preload_block_size=1000):
		self.opt = opt
		self.data_name = data_file
		self.data_file = data_file

		# when data is too large to fit in memory, it will preload a block of this many batches before calling __getitem__
		self.preload_block_size = preload_block_size

		print('loading data from {0}'.format(data_file))
		f = h5py.File(data_file, 'r')
		self.f = f

		self.batch_l = f['batch_l'][:].astype(np.int32)
		self.batch_idx = f['batch_idx'][:].astype(np.int32)
		self.ex_idx = f['ex_idx'][:].astype(np.int32)
		self.length = self.batch_l.shape[0]

		self.frame_pool = torch.from_numpy(f['frame_pool'][:].astype(np.int32))
		if opt.gpuid != -1:
			self.frame_pool = self.frame_pool.long().cuda(opt.gpuid)

		if 'semlink_map' in self.f:
			self.semlink_map = torch.from_numpy(f['semlink_map'][:].astype(np.int32))
			self.semlink_pool = torch.from_numpy(f['semlink_pool'][:].astype(np.int32))
		else:
			self.semlink_map = None
			self.semlink_pool = None

		self.f.close()

		# count examples
		self.num_ex = 0
		for i in range(self.length):
			self.num_ex += self.batch_l[i]

		# load resource files
		self.res_names = []
		if res_files is not None:
			for f in res_files:
				print('loading resource from {0}'.format(f))
				if f.endswith('txt'):
					res_names = self.__load_txt(f)

				elif f.endswith('json'):
					res_names = self.__load_json_res(f)

				elif f.endswith('frame_pool.hdf5'):
					file = h5py.File(f, 'r')
					self.frame_pool = torch.from_numpy(file['frame_pool'][:])
					res_names = ['frame_pool']

				elif f.endswith('hdf5'):
					res_names = self.__load_hdf5(f)

				else:
					assert(False)
				self.res_names.extend(res_names)


	def __preload(self, batches):
		self.batches = []
		self.batch_map = {}
		for i in tqdm(range(len(batches)), desc='preloading {0} batches'.format(len(batches))):
			b = batches[i]
			start = self.batch_idx[b]
			end = start + self.batch_l[b]

			ex_idx_i = [self.ex_idx[k] for k in range(start, end)]

			# get example token indices
			seq_l_i = self.f['seq_l'][b]
			tok_idx_i = torch.from_numpy(self.f['tok_idx'][start:end, 0:seq_l_i].astype(np.int32))
			sub2tok_idx_i = torch.from_numpy(self.f['sub2tok_idx'][start:end, 0:seq_l_i].astype(np.int32))
			v_idx_i = torch.from_numpy(self.f['v_idx'][start:end].astype(np.int32))
			v_l_i = torch.from_numpy(self.f['v_l'][start:end].astype(np.int32))
			roleset_id_i = torch.from_numpy(self.f['roleset_id'][start:end].astype(np.int32))
			roleset_suffix_i = torch.from_numpy(self.f['roleset_suffix'][start:end].astype(np.int32))
			orig_seq_l_i = torch.from_numpy(self.f['orig_seq_l'][start:end].astype(np.int32))
			srl_label_i = torch.from_numpy(self.f['srl_label'][start:end, 0:v_l_i.max(), 0:orig_seq_l_i.max()].astype(np.int32))
			frame_idx_i = torch.from_numpy(self.f['frame_idx'][start:end, 0:v_l_i.max()])
			if 'wp_idx' in self.f:
				wp_idx_i = torch.from_numpy(self.f['wp_idx'][start:end, :].astype(np.int32))
			else:
				wp_idx_i = None

			if 'wpa_idx' in self.f:
				wpa_idx_i = torch.from_numpy(self.f['wpa_idx'][start:end, :].astype(np.int32))
			else:
				wpa_idx_i = None

			if 'vn_label' in self.f:
				vn_label_i = torch.from_numpy(self.f['vn_label'][start:end, 0:v_l_i.max(), 0:orig_seq_l_i.max()].astype(np.int32))
			else:
				vn_label_i = None

			if 'vn_class' in self.f:
				vn_class_i = torch.from_numpy(self.f['vn_class'][start:end, 0:v_l_i.max()].astype(np.int32))
			else:
				vn_class_i = None

			if 'semlink_l' in self.f:
				semlink_l_i = torch.from_numpy(self.f['semlink_l'][start:end, 0:v_l_i.max()].astype(np.int32))
				semlink_i = torch.from_numpy(self.f['semlink'][start:end, 0:v_l_i.max(), :, :].astype(np.int32))
			else:
				semlink_l_i = 0
				semlink_i = None

			self.batches.append((
				ex_idx_i, tok_idx_i, wp_idx_i, wpa_idx_i, seq_l_i, orig_seq_l_i,
				sub2tok_idx_i, v_idx_i, v_l_i,
				vn_class_i, vn_label_i, srl_label_i, roleset_id_i, roleset_suffix_i,
				frame_idx_i,
				semlink_i, semlink_l_i))

			if isinstance(b, torch.Tensor):
				b = b.item()
			self.batch_map[b] = i


	def subsample(self, ratio, minimal_num=0):
		target_num_ex = int(float(self.num_ex) * ratio)
		target_num_ex = max(target_num_ex, minimal_num)
		sub_idx = torch.LongTensor(range(self.size()))
		sub_num_ex = 0

		if ratio != 1.0:
			rand_idx = torch.randperm(self.size())
			i = 0
			while sub_num_ex < target_num_ex and i < self.batch_l.shape[0]:
				sub_num_ex += self.batch_l[rand_idx[i]]
				i += 1
			sub_idx = rand_idx[:i]

		else:
			sub_num_ex = self.batch_l.sum()

		return sub_idx, sub_num_ex


	def split(self, sub_idx, ratio):
		num_ex = sum([self.batch_l[i] for i in sub_idx])
		target_num_ex = int(float(num_ex) * ratio)

		cur_num_ex = 0
		cur_pos = 0
		for i in range(len(sub_idx)):
			cur_pos = i
			cur_num_ex += self.batch_l[sub_idx[i]]
			if cur_num_ex >= target_num_ex:
				break

		return sub_idx[:cur_pos+1], sub_idx[cur_pos+1:], cur_num_ex, num_ex - cur_num_ex


	def __load_txt(self, path):
		lines = []
		# read file in unicode mode!!!
		with io.open(path, 'r+', encoding="utf-8") as f:
			for l in f:
				lines.append(l.rstrip())
		# the second last extension is the res name
		res_name = path.split('.')[-2]
		res_data = lines[:]

		# some customized parsing
		parsed = []
		if res_name == 'orig_tok_grouped':
			for l in res_data:
				parsed.append(l.rstrip().split(' '))
		else:
			parsed = res_data

		setattr(self, res_name, parsed)
		return [res_name]


	def __load_hdf5(self, path):
		res_name = path.split('.')[-2]
		file = h5py.File(path, 'r')
		setattr(self, res_name, torch.from_numpy(file[res_name][:]))
		return [res_name]


	def __load_json_res(self, path):
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		# get key name of the file
		assert(len(j_obj) == 2)
		res_type = next(iter(j_obj))

		res_name = None
		if j_obj[res_type] == 'map':
			res_name = self.__load_json_map(path)
		elif j_obj[res_type] == 'list':
			res_name = self.__load_json_list(path)
		else:
			assert(False)

		return [res_name]

	
	def __load_json_map(self, path):
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		assert(len(j_obj) == 2)

		res_name = None
		for k, v in j_obj.items():
			if k != 'type':
				res_name = k

		# optimize indices
		res = {}
		for k, v in j_obj[res_name].items():
			lut = {}
			for i, j in v.items():
				if i == res_name:
					lut[res_name] = [int(l) for l in j]
				else:
					lut[int(i)] = ([l for l in j[0]], [l for l in j[1]])

			res[int(k)] = lut
		
		setattr(self, res_name, res)
		return res_name


	def __load_json_list(self, path):
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		assert(len(j_obj) == 2)
		
		res_name = None
		for k, v in j_obj.items():
			if k != 'type':
				res_name = k

		# optimize indices
		res = {}
		for k, v in j_obj[res_name].items():
			p = v['p']
			h = v['h']

			# for token indices, shift by 1 to incorporate the nul-token at the beginning
			res[int(k)] = ([l for l in p], [l for l in h])
		
		setattr(self, res_name, res)
		return res_name


	def size(self):
		return self.length


	def __getitem__(self, idx):
		if isinstance(idx, torch.Tensor):
			idx = idx.item()

		if idx not in self.batch_map:
			start = self.batch_ls.index(idx)
			batches_to_load = self.batch_ls[start:start+self.preload_block_size]
			self.__preload(batches_to_load)

		b = self.batch_map[idx]
		(batch_ex_idx, tok_idx, wp_idx, wpa_idx, seq_l, orig_seq_l,
			sub2tok_idx, v_idx, v_l,
			vn_class, vn_label, srl_label, roleset_ids, roleset_suffixes,
			frame_idx,
			semlink, semlink_l) = self.batches[b]

		# convert all indices to long format
		tok_idx = tok_idx.long()
		sub2tok_idx = sub2tok_idx.long()
		v_idx = v_idx.long()	
		srl_label = srl_label.long()
		roleset_ids = roleset_ids.long()
		roleset_suffixes = roleset_suffixes.long()
		frame_idx = frame_idx.long()
		if wp_idx is not None:
			wp_idx = wp_idx.long()
		if wpa_idx is not None:
			wpa_idx = wpa_idx.long()
		if vn_label is not None:
			vn_label = vn_label.long()
		if vn_class is not None:
			vn_class = vn_class.long()
		if semlink is not None:
			semlink = semlink.long()
			semlink_l = semlink_l.long()

		# transfer to gpu if needed
		if self.opt.gpuid != -1:
			tok_idx = tok_idx.cuda(self.opt.gpuid)
			sub2tok_idx = sub2tok_idx.cuda(self.opt.gpuid)
			v_idx = v_idx.cuda(self.opt.gpuid)
			srl_label = srl_label.cuda(self.opt.gpuid)
			roleset_ids = roleset_ids.cuda(self.opt.gpuid)
			roleset_suffixes = roleset_suffixes.cuda(self.opt.gpuid)
			frame_idx = frame_idx.cuda(self.opt.gpuid)
			if wp_idx is not None:
				wp_idx = wp_idx.cuda(self.opt.gpuid)
			if wpa_idx is not None:
				wpa_idx = wpa_idx.cuda(self.opt.gpuid)
			if vn_label is not None:
				vn_label = vn_label.cuda(self.opt.gpuid)
			if vn_class is not None:
				vn_class = vn_class.cuda(self.opt.gpuid)
			if semlink is not None:
				semlink = semlink.cuda(self.opt.gpuid)
				semlink_l = semlink_l.cuda(self.opt.gpuid)

		# get batch ex indices
		res_map = self.__get_res(idx)

		batch = Holder()
		batch.data_name = self.data_name
		batch.tok_idx = tok_idx
		batch.wp_idx = wp_idx
		batch.wpa_idx = wpa_idx
		batch.batch_ex_idx = batch_ex_idx
		batch.batch_l = self.batch_l[idx]
		batch.seq_l = seq_l
		batch.orig_seq_l = orig_seq_l
		batch.sub2tok_idx = sub2tok_idx
		batch.v_label = v_idx
		batch.v_l = v_l
		batch.vn_class = vn_class
		batch.vn_label = vn_label
		batch.srl_label = srl_label
		batch.roleset_ids = roleset_ids
		batch.roleset_suffixes = roleset_suffixes
		batch.frame_idx = frame_idx
		batch.frame_pool = self.frame_pool
		batch.semlink = semlink
		batch.semlink_l = semlink_l
		batch.res_map = res_map

		return batch
		

	def __get_res(self, idx):
		# if there is no resource presents, return None
		if len(self.res_names) == 0:
			return None

		b = self.batch_map[idx]
		batch_ex_idx = self.batches[b][0]

		all_res = {}
		for res_n in self.res_names:
			res = getattr(self, res_n)
			if res_n == 'frame_pool':	# frame_pool is invariant across examples
				all_res[res_n] = self.frame_pool
			elif type(res) == list:
				batch_res = [res[ex_id] for ex_id in batch_ex_idx]
				all_res[res_n] = batch_res
			else:
				all_res[res_n] = res[batch_ex_idx]

		return all_res


	# something at the beginning of each pass of training/eval
	#	e.g. setup preloading
	def begin_pass(self, batch_ls):
		self.batches = []
		self.batch_map = {}
		self.batch_ls = batch_ls
		if isinstance(self.batch_ls, torch.Tensor):
			self.batch_ls = self.batch_ls.tolist()
		self.f = h5py.File(self.data_file, 'r')

	def end_pass(self):
		self.f.close()
