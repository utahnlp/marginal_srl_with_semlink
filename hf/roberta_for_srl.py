import torch
import transformers
from transformers import *
from .loss import *
from .encoder import *
from .classifier import *
from .util import *
from packaging import version
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers.models.roberta.configuration_roberta import RobertaConfig
import nltk
from nltk.tokenize import TreebankWordTokenizer
tb_tokenizer = TreebankWordTokenizer()


def process_options_(config):
	if hasattr(config, 'arg_role_map_inv'):
		config.arg_role_map_inv = {int(k): v for k, v in config.arg_role_map_inv.items()}
	if hasattr(config, 'condensed_map_inv'):
		config.condensed_map_inv = {int(k): v for k, v in config.condensed_map_inv.items()}
	if hasattr(config, 'roleset_map_inv'):
		config.roleset_map_inv = {int(k): v for k, v in config.roleset_map_inv.items()}
	if hasattr(config, 'vn_class_map_inv'):
		config.vn_class_map_inv = {int(k): v for k, v in config.vn_class_map_inv.items()}
	if hasattr(config, 'vn_label_map_inv'):
		config.vn_label_map_inv = {int(k): v for k, v in config.vn_label_map_inv.items()}
	if hasattr(config, 'semlink'):
		config.semlink = {tuple(k.split('/')): v for k, v in config.semlink.items()}
	if hasattr(config, 'semlink_map'):
		config.semlink_map = np.asarray(config.semlink_map, dtype=np.int32)


def preprocess_input(config, tokenizer, word_toks, v_idx, vnclasses, senses):
	"""
	Preprocess an input before feeding to a model.
	Args:
		config: the global configuration.
		word_toks: word-tokenized sentence, i.e., a list of words.
		v_idx: predicate locations in word_toks, i.e., a list of integers.
		vnclasses: VerbNet classes of predicates, must be the same length as v_idx.
		senses: Roleset ids of predicates, must be the same length as v_idx.
	Returns:
		a pack of processed features.
	"""
	orig_toks = word_toks

	sent_subtoks = [tokenizer.tokenize(t) for t in orig_toks]
	tok_l = [len(subtoks) for subtoks in sent_subtoks]
	toks = [p for subtoks in sent_subtoks for p in subtoks]	# flatterning

	# pad for CLS and SEP
	CLS, SEP = tokenizer.cls_token, tokenizer.sep_token
	toks = [CLS] + toks + [SEP]
	tok_l = [1] + tok_l + [1]
	orig_toks = [CLS] + orig_toks + [SEP]
	v_idx = [l+1 for l in v_idx]

	tok_idx = np.array(tokenizer.convert_tokens_to_ids(toks), dtype=int)

	# note that the resulted actual seq length after subtoken collapsing could be different even within the same batch
	#	actual seq length is the origial sequence length
	#	seq length is the length after subword tokenization
	acc = 0
	sub2tok_idx = []
	for l in tok_l:
		sub2tok_idx.append(pad([p for p in range(acc, acc+l)], config.max_num_subtok, -1))
		assert(len(sub2tok_idx[-1]) <= config.max_num_subtok)
		acc += l
	sub2tok_idx = pad(sub2tok_idx, len(tok_idx), [-1 for _ in range(config.max_num_subtok)])
	sub2tok_idx = np.array(sub2tok_idx, dtype=int)

	roleset_id = [config.roleset.index(p) if p in config.roleset else 0 for p in senses]
	suffixes = [p.split('.')[-1] for p in senses]
	roleset_suffix = [config.roleset_suffix.index(p) if p in config.roleset_suffix else 0 for p in suffixes]
	vn_class = [config.vn_classes.index(p) if p in config.vn_classes else 0 for p in vnclasses]

	roleset_toks = [tokenizer.tokenize(sense) for sense in senses]
	roleset_toks = [p for subtoks in roleset_toks for p in subtoks]
	roleset_toks = roleset_toks + [tokenizer.sep_token]
	wp_idx = np.array(tokenizer.convert_tokens_to_ids(roleset_toks), dtype=int)

	semlink = np.zeros((len(vnclasses), 2, config.max_num_semlink), dtype=np.int32) - 1
	semlink_l = [0 for _ in vnclasses]
	for k, (vnc_id, sense_id) in enumerate(zip(vn_class, roleset_suffix)):
		semlink_alignments = config.semlink_map[vnc_id, sense_id]
		semlink_l = (semlink_alignments[0] != -1).sum()
		semlink[k] = semlink_alignments

	orig_seq_l = [len(orig_toks) for _ in v_idx]
	# batch size 1
	batch = Holder()
	batch.toks = toks
	batch.orig_toks = torch.tensor([tok_idx], dtype=torch.long)
	batch.tok_idx = torch.tensor([tok_idx], dtype=torch.long)
	batch.seq_l = tok_idx.shape[-1]
	batch.orig_seq_l = torch.tensor(orig_seq_l, dtype=torch.long)
	batch.sub2tok_idx = torch.tensor([sub2tok_idx], dtype=torch.long)
	batch.wp_idx = torch.tensor([wp_idx], dtype=torch.long)
	batch.v_label = torch.tensor([v_idx], dtype=torch.long)
	batch.v_l = torch.tensor([len(v_idx)], dtype=torch.long)
	batch.vn_class = torch.tensor([vn_class], dtype=torch.long)
	batch.roleset_ids = torch.tensor([roleset_id], dtype=torch.long)
	batch.roleset_suffixes = torch.tensor([roleset_suffix], dtype=torch.long)
	batch.semlink = torch.tensor([semlink], dtype=torch.long)
	batch.semlink_l = torch.tensor([semlink_l], dtype=torch.long)
	return batch


def batch_to_device(batch, device):
	batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
	rs = Holder()
	rs.update(batch)
	return rs


# RoBERTa for Marginal SRL, only meant for inference/demo
class RobertaForSRL(RobertaPreTrainedModel):
	authorized_unexpected_keys = [r"pooler"]
	authorized_missing_keys = [r"position_ids"]

	def __init__(self, config, *model_args, **model_kwargs):
		super().__init__(config)

		# comment this out when calling convert_checkpoint.py
		process_options_(config)

		self.shared = Holder()

		# these are placeholders.
		self.encoder = WPEncoder(config, self.shared)
		self.classifier = MergeConcatClassifier(config, self.shared)
		self.loss = CrossCRFLoss(config, self.shared)

	def _update_context(self, batch):
		# For now, only one example
		self.shared.batch_ex_idx = torch.zeros((1,), dtype=torch.long)
		self.shared.batch_l = 1
		self.shared.update(batch)
		self.shared.res_map = {}

	# shared: a namespace or a Holder instance that contains information for the current input batch
	#	such as, predicate labels, subtok to tok index mapping, etc
	def classify(self, batch):
		self._update_context(batch)
		# encoder
		enc = self.encoder(batch)

		# classifier
		cls_pack = self.classifier(enc)

		outputs = self.loss(cls_pack)
		return outputs


if __name__ == '__main__':
	tokenizer = AutoTokenizer.from_pretrained('tli8hf/roberta-base-marginal-semlink')
	model = RobertaForSRL.from_pretrained('tli8hf/roberta-base-marginal-semlink')
	model = model.to('cuda:0')
	config = model.config

	orig_toks = 'The company said it will continue to pursue a lifting of the suspension .'.split()
	v_idx = [2, 7]
	vnclasses = ['B-37.7', 'B-35.6']
	senses = ['say.01', 'pursue.01']
	batch = preprocess_input(config, tokenizer, orig_toks, v_idx, vnclasses, senses)
	batch = batch_to_device(batch, 'cuda:0')
	outputs = model.classify(batch)
	print(outputs)
