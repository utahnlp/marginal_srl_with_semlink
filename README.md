


-----

## Preprocessing

### 1. First install dependencies.
```
pip install -r requirements.txt
```

### 2. Data extraction and processing
```
scripts/semlink/semlink2conll.sh --ptb [PATH_TO_TREEBANK3_PARSED_MRG_DIR] \
--roleset both \
--senses vn \
--brown data/datasets/conll-brown-release/prop.txt

scripts/semlink/semlink2conll_brown.sh --ptb [PATH_TO_TREEBANK3_PARSED_MRG_DIR] \
--roleset both \
--senses vn \
--brown data/datasets/conll-brown-release/prop.txt
```
Specify the `PATH_TO_TREEBANK3_PARSED_MRG_DIR` to the right directory.

### 3. Further extraction and processing

Follow [iwcs2021](https://github.com/jgung/verbnet-parsing-iwcs-2021) to extract the joint set of VN and PB.
Let `PATH_TO_IWCS2021_REPO` point to the cloned repository in your system.

Then follow [structured_tuning_2021](https://github.com/utahnlp/structured_tuning_srl) to process CoNLL 05 dataset. We will need those to continue the extraction.
Let `CONLL05_DIR` points to the processed directory


```
CONLL_PATH=PATH_TO_IWCS2021_REPO/semlink1.1
python3 process_iwcs2021.py ${CONLL_PATH}/vn.both-valid.txt \
	${CONLL_PATH}/dev.txt \
	${CONLL_PATH}/dev.props.gold.txt \
 	${CONLL_PATH}/dev.propid.txt \
 	${CONLL_PATH}/dev.fileid.txt \
 	${CONLL_PATH}/dev.origfileid.txt

python3 process_iwcs2021.py ${CONLL_PATH}/vn.both-train.txt \
	${CONLL_PATH}/train.txt \
	${CONLL_PATH}/train.props.gold.txt \
 	${CONLL_PATH}/train.propid.txt \
 	${CONLL_PATH}/train.fileid.txt \
 	${CONLL_PATH}/train.origfileid.txt

python3 process_iwcs2021.py ${CONLL_PATH}/vn.both-test-wsj.txt \
	${CONLL_PATH}/test1.txt \
	${CONLL_PATH}/test1.props.gold.txt \
 	${CONLL_PATH}/test1.propid.txt \
 	${CONLL_PATH}/test1.fileid.txt \
 	${CONLL_PATH}/test1.origfileid.txt

python3 process_iwcs2021.py ${CONLL_PATH}/vn.both-test-brown.txt \
	${CONLL_PATH}/test2.txt \
	${CONLL_PATH}/test2.props.gold.txt \
 	${CONLL_PATH}/test2.propid.txt \
 	${CONLL_PATH}/test2.fileid.txt \
 	${CONLL_PATH}/test2.origfileid.txt
```


### 4. Final processing to batch up examples

```
python3 -u -m preprocess.extract_iwcs2021 | tee extract.iwcs2021.log.txt
python3 -u -m preprocess.preprocess_iwcs2021 --output iwcs_wp | tee preprocess.iwcs2021.log.txt
python3 -u -m preprocess.extract_extra_srl
python3 -u -m preprocess.preprocess_extra_srl \
	--arg_role_map iwcs_wp.arg_role.txt \
	--indexer iwcs_wp \
	--output iwcs_wp.extra.conll05


python3 -u -m preprocess.extract_extra_srl --data ./conll05.test.wsj.txt --filter "" --output iwcs.entire.conll05.test1.vn_srl.txt
python3 -u -m preprocess.extract_extra_srl --data ./conll05.test.brown.txt --filter "" --output iwcs.entire.conll05.test2.vn_srl.txt
python3 -u -m preprocess.preprocess_extra_srl \
	--data iwcs.entire.conll05.test2.vn_srl.txt \
	--arg_role_map iwcs_wp.arg_role.txt \
	--indexer iwcs_wp.extra.conll05 \
	--aug_indexer 0 \
	--output iwcs_wp.entire.conll05.test2


python3 -u -m preprocess.preprocess_iwcs2021_completion --output iwcs_completion
python3 -u -m preprocess.preprocess_extra_srl_completion \
	--arg_role_map iwcs_completion.arg_role.txt \
	--indexer iwcs_completion \
	--output iwcs_completion.extra.conll05
```


## Training

### 1. Train a joint model for the 1st round
```
for SEED in 1 2 3; do
EP=20
DROP=0.5
LR=0.00003
DATA=iwcs_wp
SIZE=base
ENC=bert_wp
LAMBD=1,1
MODEL=./models/multi_${DATA}_${SIZE}_${ENC//_}_ep${EP}_lr${LR//.}_drop${DROP//.}_multiw${LAMBD//,}_seed${SEED}
python3 -u train.py --gpuid [GPUID] --dir [DATA] \
	--train_data ${DATA}.train.hdf5 --val_data ${DATA}.val.hdf5 \
	--train_res ${DATA}.train.orig_tok_grouped.txt --val_res ${DATA}.val.orig_tok_grouped.txt \
	--data_flag vn_srl \
	--vn_class_dict ${DATA}.vn_class.dict \
	--vn_label_dict ${DATA}.vn_label.dict --srl_label_dict ${DATA}.srl_label.dict \
	--roleset_dict ${DATA}.roleset_label.dict --roleset_suffix_dict ${DATA}.roleset_suffix.dict \
	--arg_role_map ${DATA}.arg_role.txt \
	--enc $ENC --cls multi --loss multi_crf --multi_w ${LAMBD} \
	--optim adamw_fp16 --epochs $EP --learning_rate $LR --dropout $DROP \
	--bert_type roberta-${SIZE} --hidden_size 768 \
	--percent 1 --val_percent 1 \
	--seed $SEED --conll_output $MODEL --save_file $MODEL | tee ${MODEL}.txt
done
```
Specify the `[GPUID]` and `[DATA]`.

### 2. Train for the 2nd round
```
for SEED in 1 2 3; do
EP=5
DROP=0.5
LR=0.00001
DATA=iwcs_wp
SIZE=base
ENC=bert_wp
LAMBD=1,1
LOAD=./models/multi_${DATA}_${SIZE}_${ENC//_}_ep20_lr000003_drop${DROP//.}_multiw${LAMBD//,}_seed${SEED}
MODEL=./models/multi2_${DATA}_${SIZE}_${ENC//_}_ep${EP}_lr${LR//.}_drop${DROP//.}_multiw${LAMBD//,}_seed${SEED}
python3 -u train.py --gpuid [GPUID] --dir [DATA] \
	--load_file $LOAD \
	--train_data ${DATA}.train.hdf5 --val_data ${DATA}.val.hdf5 \
	--train_res ${DATA}.train.orig_tok_grouped.txt --val_res ${DATA}.val.orig_tok_grouped.txt \
	--data_flag vn_srl \
	--vn_class_dict ${DATA}.vn_class.dict \
	--vn_label_dict ${DATA}.vn_label.dict --srl_label_dict ${DATA}.srl_label.dict \
	--roleset_dict ${DATA}.roleset_label.dict --roleset_suffix_dict ${DATA}.roleset_suffix.dict \
	--arg_role_map ${DATA}.arg_role.txt \
	--enc $ENC --cls multi --loss multi_crf --multi_w ${LAMBD} \
	--optim adamw_fp16 --epochs $EP --learning_rate $LR --dropout $DROP \
	--bert_type roberta-${SIZE} --hidden_size 768 \
	--percent 1 --val_percent 1 \
	--seed $SEED --conll_output $MODEL --save_file $MODEL | tee ${MODEL}.txt
done
```
Specify the `[GPUID]` and `[DATA]`.

To evaluate the model, run:
```
for SEED in 1 2 3; do
for SPLIT in test1 test2; do
EP=5
DROP=0.5
LR=0.00001
SIZE=base
ENC=bert_wp
DATA=iwcs_wp
LAMBD=1,1
EVAL_SRL=1
MODEL=./models/multi2_${DATA}_${SIZE}_${ENC//_}_ep${EP}_lr${LR//.}_drop${DROP//.}_multiw${LAMBD//,}_seed${SEED}
python3 -u eval.py --gpuid [GPUID] --dir [DATA] \
	--data ${DATA}.${SPLIT}.hdf5 \
	--res ${DATA}.${SPLIT}.orig_tok_grouped.txt \
	--data_flag vn_srl \
	--vn_class_dict ${DATA}.vn_class.dict \
	--vn_label_dict ${DATA}.vn_label.dict --srl_label_dict ${DATA}.srl_label.dict \
	--roleset_dict ${DATA}.roleset_label.dict --roleset_suffix_dict ${DATA}.roleset_suffix.dict \
	--arg_role_map ${DATA}.arg_role.txt \
	--enc $ENC --cls multi --loss multi_crf --multi_w ${LAMBD} --eval_with_srl ${EVAL_SRL} \
	--bert_type roberta-${SIZE} \
	--seed $SEED --conll_output ${MODEL} --log pretty --load_file $MODEL | tee ${MODEL}.${SPLIT}.txt
done
done
```
Specify the `[GPUID]` and `[DATA]`.