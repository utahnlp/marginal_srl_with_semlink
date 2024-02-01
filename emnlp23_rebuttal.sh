

# multi model
for SPLIT in val; do
SIZE=base
ENC=bert_wp
DATA=iwcs_wp
LAMBD=1,1
EVAL_SRL=0
MODEL=./models/multi2_iwcs_wp_base_bertwp_ep5_lr000001_drop05_multiw11_seed1
python3 -u eval.py --gpuid 0 --dir ./data/vn_modular/ \
	--data ${DATA}.${SPLIT}.hdf5 \
	--res ${DATA}.${SPLIT}.orig_tok_grouped.txt \
	--data_flag vn_srl \
	--vn_class_dict ${DATA}.vn_class.dict \
	--vn_label_dict ${DATA}.vn_label.dict --srl_label_dict ${DATA}.srl_label.dict \
	--roleset_dict ${DATA}.roleset_label.dict --roleset_suffix_dict ${DATA}.roleset_suffix.dict \
	--arg_role_map ${DATA}.arg_role.txt \
	--enc $ENC --cls multi --loss multi_crf --multi_w ${LAMBD} --eval_with_srl ${EVAL_SRL} \
	--bert_type roberta-${SIZE} \
	--seed 1 --conll_output ${MODEL} --log pretty --load_file $MODEL
done


# joint model
for SPLIT in val; do
for EVAL_SEMLINK in 1; do
SIZE=base
ENC=bert_wp
DATA=iwcs_wp
SEMLINK=1
EVAL_FRAME=0
EVAL_SRL=0
MODEL=./models/pass2_iwcs_wp_base_bertwp_semlink1_ep5_lr000001_drop05_seed1
python3 -u eval.py --gpuid 0 --dir ./data/vn_modular/ \
	--data ${DATA}.${SPLIT}.hdf5 \
	--res ${DATA}.${SPLIT}.orig_tok_grouped.txt \
	--data_flag cross_semlink \
	--vn_class_dict ${DATA}.vn_class.dict \
	--vn_label_dict ${DATA}.vn_label.dict --srl_label_dict ${DATA}.srl_label.dict \
	--roleset_dict ${DATA}.roleset_label.dict --roleset_suffix_dict ${DATA}.roleset_suffix.dict \
	--arg_role_map ${DATA}.arg_role.txt \
	--enc $ENC --cls merge_concat --loss cross_crf \
	--eval_with_semlink ${EVAL_SEMLINK} --eval_with_frame ${EVAL_FRAME} --eval_with_srl ${EVAL_SRL} \
	--bert_type roberta-${SIZE} \
	--seed 1 --conll_output ${MODEL}_evalsemlink${EVAL_SEMLINK}_evalframe${EVAL_FRAME} --log pretty --load_file $MODEL
done
done



# semi models
for SPLIT in val; do
for EVAL_SEMLINK in 1; do
SIZE=base
ENC=bert_wp
DATA=iwcs_wp
EXTRA=iwcs_wp.extra.conll05
SEMLINK=1
EVAL_FRAME=0
EVAL_SRL=0
MODEL=./models/semi2_marginalsrl_iwcs_wp_base_bertwp_semlink1_ep5_lr0000003_drop05_seed1
python3 -u eval.py --gpuid 0 --dir ./data/vn_modular/ \
	--data ${DATA}.${SPLIT}.hdf5 \
	--res ${DATA}.${SPLIT}.orig_tok_grouped.txt \
	--data_flag cross \
	--vn_class_dict ${EXTRA}.vn_class.dict \
	--vn_label_dict ${EXTRA}.vn_label.dict --srl_label_dict ${EXTRA}.srl_label.dict \
	--roleset_dict ${EXTRA}.roleset_label.dict --roleset_suffix_dict ${EXTRA}.roleset_suffix.dict \
	--arg_role_map ${DATA}.arg_role.txt \
	--enc $ENC --cls merge_concat --loss cross_crf \
	--eval_with_semlink ${EVAL_SEMLINK} --eval_with_frame ${EVAL_FRAME} --eval_with_srl ${EVAL_SRL} \
	--bert_type roberta-${SIZE} \
	--seed 1 --conll_output ${MODEL}_evalsemlink${EVAL_SEMLINK}_evalframe${EVAL_FRAME} --log pretty --load_file $MODEL
done
done


for SPLIT in val; do
for EVAL_SEMLINK in 1; do
SIZE=base
ENC=bert_wp
DATA=iwcs_wp
EXTRA=iwcs_wp.extra.conll05
SEMLINK=1
EVAL_FRAME=0
EVAL_SRL=0
MODEL=./models/semi2_marginalsrlsemlinklocal_iwcs_wp_base_bertwp_semlink1_ep5_lr0000001_drop05_seed1
python3 -u eval.py --gpuid 0 --dir ./data/vn_modular/ \
	--data ${DATA}.${SPLIT}.hdf5 \
	--res ${DATA}.${SPLIT}.orig_tok_grouped.txt \
	--data_flag cross \
	--vn_class_dict ${EXTRA}.vn_class.dict \
	--vn_label_dict ${EXTRA}.vn_label.dict --srl_label_dict ${EXTRA}.srl_label.dict \
	--roleset_dict ${EXTRA}.roleset_label.dict --roleset_suffix_dict ${EXTRA}.roleset_suffix.dict \
	--arg_role_map ${DATA}.arg_role.txt \
	--enc $ENC --cls merge_concat --loss cross_crf \
	--eval_with_semlink ${EVAL_SEMLINK} --eval_with_frame ${EVAL_FRAME} --eval_with_srl ${EVAL_SRL} \
	--bert_type roberta-${SIZE} \
	--seed 1 --conll_output ${MODEL}_evalsemlink${EVAL_SEMLINK}_evalframe${EVAL_FRAME} --log pretty --load_file $MODEL
done
done




----------------------- corruption on the val

python3 -u -m preprocess.corrupt_predicate_iwcs2021 --indexer iwcs_wp.extra.conll05 --output iwcs_wp_corrupted_vn005 --corrupt_type vnclass --probability 0.05
python3 -u -m preprocess.corrupt_predicate_iwcs2021 --indexer iwcs_wp.extra.conll05 --output iwcs_wp_corrupted_vn01 --corrupt_type vnclass --probability 0.1
python3 -u -m preprocess.corrupt_predicate_iwcs2021 --indexer iwcs_wp.extra.conll05 --output iwcs_wp_corrupted_vn02 --corrupt_type vnclass --probability 0.2
python3 -u -m preprocess.corrupt_predicate_iwcs2021 --indexer iwcs_wp.extra.conll05 --output iwcs_wp_corrupted_vn03 --corrupt_type vnclass --probability 0.3

python3 -u -m preprocess.corrupt_predicate_iwcs2021 --indexer iwcs_wp.extra.conll05 --output iwcs_wp_corrupted_srl005 --corrupt_type sense --probability 0.05
python3 -u -m preprocess.corrupt_predicate_iwcs2021 --indexer iwcs_wp.extra.conll05 --output iwcs_wp_corrupted_srl01 --corrupt_type sense --probability 0.1
python3 -u -m preprocess.corrupt_predicate_iwcs2021 --indexer iwcs_wp.extra.conll05 --output iwcs_wp_corrupted_srl02 --corrupt_type sense --probability 0.2
python3 -u -m preprocess.corrupt_predicate_iwcs2021 --indexer iwcs_wp.extra.conll05 --output iwcs_wp_corrupted_srl03 --corrupt_type sense --probability 0.3

for SPLIT in val; do
for EVAL_SEMLINK in 1; do
SIZE=base
ENC=bert_wp
DATA=iwcs_wp_corrupted_srl01
EXTRA=iwcs_wp.extra.conll05
SEMLINK=1
EVAL_FRAME=0
EVAL_SRL=0
MODEL=./models/semi2_marginalsrlsemlinklocal_iwcs_wp_base_bertwp_semlink1_ep5_lr0000001_drop05_seed1
python3 -u eval.py --gpuid 0 --dir ./data/vn_modular/ \
	--data ${DATA}.${SPLIT}.hdf5 \
	--res ${DATA}.${SPLIT}.orig_tok_grouped.txt \
	--data_flag cross \
	--vn_class_dict ${EXTRA}.vn_class.dict \
	--vn_label_dict ${EXTRA}.vn_label.dict --srl_label_dict ${EXTRA}.srl_label.dict \
	--roleset_dict ${EXTRA}.roleset_label.dict --roleset_suffix_dict ${EXTRA}.roleset_suffix.dict \
	--arg_role_map iwcs_wp.arg_role.txt \
	--enc $ENC --cls merge_concat --loss cross_crf \
	--eval_with_semlink ${EVAL_SEMLINK} --eval_with_frame ${EVAL_FRAME} --eval_with_srl ${EVAL_SRL} \
	--bert_type roberta-${SIZE} \
	--seed 1 --conll_output ${MODEL}_evalsemlink${EVAL_SEMLINK}_evalframe${EVAL_FRAME} --log pretty --load_file $MODEL

	VN_REF='./models/pass2_iwcs_wp_base_bertwp_semlink1_ep5_lr000001_drop05_seed1_evalsemlink1_evalframe0.cross_crf_vn_gold.txt'
	SRL_REF='./models/pass2_iwcs_wp_base_bertwp_semlink1_ep5_lr000001_drop05_seed1_evalsemlink1_evalframe0.cross_crf_srl_gold.txt'

	VN_PRED='./models/semi2_marginalsrlsemlinklocal_iwcs_wp_base_bertwp_semlink1_ep5_lr0000001_drop05_seed1_evalsemlink1_evalframe0.cross_crf_vn_pred.txt'
	SRL_PRED='./models/semi2_marginalsrlsemlinklocal_iwcs_wp_base_bertwp_semlink1_ep5_lr0000001_drop05_seed1_evalsemlink1_evalframe0.cross_crf_srl_pred.txt'

	perl srl-eval.pl ${VN_REF} ${VN_PRED}

	perl srl-eval.pl ${SRL_REF} ${SRL_PRED}
done
done

