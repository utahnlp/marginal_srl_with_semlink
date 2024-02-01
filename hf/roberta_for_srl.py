import torch
import transformers
from transformers import *
from loss.crf_loss import *
from loss.vnclass_loss import *
from loss.sense_loss import *
from modules.vn_linear_classifier import *
from modules.srl_linear_classifier import *
from util.util import *
from packaging import version
if version.parse(transformers.__version__) < version.parse('4.0'):
	# for transformers 3+
	from transformers.modeling_roberta import RobertaPreTrainedModel
	from transformers.configuration_roberta import RobertaConfig
else:
	# for transformers 4+
	from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
	from transformers.models.roberta.configuration_roberta import RobertaConfig


# RoBERTa for SRL, only meant for inference/demo
#	Training interface is not maintained here
class RobertaForSRL(RobertaPreTrainedModel):
	authorized_unexpected_keys = [r"pooler"]
	authorized_missing_keys = [r"position_ids"]

	def __init__(self, config, *model_args, **model_kwargs):
		super().__init__(config)

		# the config and global opt should be handled better here, for now it's hacky
		# options can be overwritten by externally specified ones
		if 'global_opt' in model_kwargs:
			for k, v in model_kwargs['global_opt'].__dict__.items():
				setattr(config, k, v)
			for k, v in config.__dict__.items():
				setattr(model_kwargs['global_opt'], k, v)
		# explicitly trigger ad-hoc fix for options
		#	some options will have wrong data type after being loaded from config.json
		complete_opt(config)
		if 'global_opt' in model_kwargs:
			complete_opt(model_kwargs['global_opt'])

		self.num_labels = config.num_labels

		self.shared = model_kwargs['shared']
		self._loss_context = Holder()

		# these are placeholders.
		self.roberta = RobertaModel(config)
		if config.cls == 'vn_linear':
			self.classifier = VNLinearClassifier(config, shared=self.shared)
		elif config.cls == 'srl_linear':
			self.classifier = SRLLinearClassifier(config, shared=self.shared)
		else:
			raise Exception('unrecognized config.cls', config.cls)

		if config.loss[0] == 'vn_crf' or config.loss[0] == 'srl_crf':
			self.crf_loss = CRFLoss(config.loss[0], opt=config, shared=self.shared)
		elif config.loss[0] == 'vnclass':
			self.v_class_loss = VNClassLoss(name='vnclass', opt=config, shared=self.shared)
		elif config.loss[0] == 'sense':
			self.v_class_loss = SenseLoss(name='sense', opt=config, shared=self.shared)

		self.init_weights()

	# update the contextual info of current batch
	def update_context(self, orig_seq_l, sub2tok_idx, res_map=None):
		self.shared.orig_seq_l = orig_seq_l
		self.shared.sub2tok_idx = sub2tok_idx
		self.shared.res_map = res_map

	# shared: a namespace or a Holder instance that contains information for the current input batch
	#	such as, predicate labels, subtok to tok index mapping, etc
	def forward(self, input_ids, v_idx, v_l, do_crf=True):
		self.shared.batch_l = input_ids.shape[0]
		self.shared.seq_l = input_ids.shape[1]

		enc = self.roberta(input_ids)[0]

		pack = self.classifier(enc)

		if do_crf:
			pred, _ = self.crf_loss.decode(pack, v_idx, v_l)
			pack['crf_pred'] = pred

		return pack