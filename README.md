# Multi-Sentence Resampling
### This is the official repository for EMNLP 2021 conference paper: ["Multi-Sentence Resampling: A Simple Approach to Alleviate Dataset Length Bias and Beam-Search Degradation"](https://aclanthology.org/2021.emnlp-main.677/)


### Installation

```
# Conda environment
conda env create -n msr_env python=3.7 -f environment.yaml

# We use fairseq 10.0
cd fairseq
pip install --editable ./
```

### resample_raw_dataset.py

This is a python file that can build a new training dataset with Multi-Sentence Resampling.

Example run:

```
# Here we set max number of concatenated sentences to 4, specify paths to training data, 
# and set number of training examples in the new training dataset to be 
# 3 times more then in the original dataset
# This file also provides opportunity to make weighted sampling, 
# where probability of a sentence is proportional to its length (default is --not-weighted).
maxsentences=4
python3 resample_raw_dataset.py\
    --source-path ${SOURCE_DIR}/train.tok.clean.bpe.32000.en\
    --target-path ${SOURCE_DIR}/train.tok.clean.bpe.32000.de\
    --directory ${TARGET_DIR}\
    --dataset-size-multiplier 3\
    --not-weighted\ 
    --silent\
    --max-sentences $maxsentences
```

### Data

You can download [gzip archive](https://www.dropbox.com/s/3cpk31n5fsbht2e/msr_data.tar.gz?dl=0) that contains checkpoints for each direction, evaluation scripts and preprocessed datasets that we used.


### Data preparation, training and evaluation

Here is an example script with several steps:
1) Creation of the new training dataset using **Multi-Sentence Resampling with resample_raw_dataset.py**
2) Prepare mmap dataset
3) Training procedure
4) Generate results for different beam-sizes

* We run it on 8 GPU.
* It uses WMT En-De dataset.
* You can use parts of this script as you want.

```
MODEL_NAME='model'
# Make MSR dataset with resample_raw_dataset.py

SOURCE_DIR=${INPUT_PATH}
TARGET_DIR=${INPUT_PATH}/resampled_wmt/
mkdir $TARGET_DIR;
cp $SOURCE_DIR/* $TARGET_DIR;

# Here we set max number of concatenated sentences to 4, specify paths to training data, 
# and set number of training examples in the new training dataset to be 
# 3 times more then in the original dataset
maxsentences=4
python3 resample_raw_dataset.py\
    --source-path ${SOURCE_DIR}/train.tok.clean.bpe.32000.en\
    --target-path ${SOURCE_DIR}/train.tok.clean.bpe.32000.de\
    --directory ${TARGET_DIR}\
    --dataset-size-multiplier 3\
    --not-weighted\
    --silent\
    --max-sentences $maxsentences

# Make mmap dataset
TEXT=${TARGET_DIR}
DST=${INPUT_PATH}/dataset_mmap/
mkdir DST;

fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train.tok.clean.bpe.32000 \
    --validpref $TEXT/newstest2013.tok.bpe.32000 \
    --testpref $TEXT/newstest2014.tok.bpe.32000 \
    --destdir $DST \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary \
    --workers 20

cp -r ${DST} ${LOGS_PATH}/

# Training script

# Dir with data and vocs
BIN_DIR=${DST}
TENSORBOARD_LOGDIR=$LOGS_PATH/temp
SAVEDIR=$LOGS_PATH/checkpoints

SRC_LANG='en'
TGT_LANG='de'
mkdir $SAVEDIR;
mkdir $TENSORBOARD_LOGDIR;

fairseq-train \
    $BIN_DIR \
    --arch transformer --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.1 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4000 \
    --fp16\
    --bpe subword_nmt\
    --tokenizer moses\
    --source-lang $SRC_LANG \
    --target-lang $TGT_LANG \
    --eval-bleu \
    --eval-bleu-args '{"beam": 10, "normalize_scores": "False"}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe="subword_nmt" \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric\
    --save-interval 1\
    --keep-best-checkpoints 5\
    --keep-last-epochs 10\
    --max-update 150000\
    --tensorboard-logdir $TENSORBOARD_LOGDIR\
    --save-dir $SAVEDIR\
    --seed 101


# Checkpoint averaging
CHECKPOINT_FILENAME=avg_last_5_checkpoint.pt
SAVE_DIR=${SAVEDIR}
BLEU_CHECKPOINTS=$(ls -d ${SAVE_DIR}/checkpoint[^._]* | sort | tail -n 5)

python3 scripts/average_checkpoints.py \
    --inputs ${BLEU_CHECKPOINTS}\
    --output "${SAVE_DIR}/${CHECKPOINT_FILENAME}"

# Generate results for different beam sizes
CHECKPOINT=${SAVE_DIR}/${CHECKPOINT_FILENAME}

RESULT_PATH=${LOGS_PATH}/normalized/
mkdir $RESULT_PATH

for beam in 5 10 20 40 60 80 100 
do
    echo ${beam}
    fairseq-generate ${BIN_DIR} \
        --task translation\
        --path ${CHECKPOINT} \
        --gen-subset 'test'\
        --batch-size 8 --beam ${beam}\
        --remove-bpe\
        --tokenizer moses\
        --source-lang $SRC_LANG \
        --target-lang $TGT_LANG \
        --scoring sacrebleu\
        > ${RESULT_PATH}generated_${beam}.txt;
done

for beam in 400 800
do
    echo ${beam}
    fairseq-generate ${BIN_DIR} \
        --task translation\
        --path ${CHECKPOINT} \
        --gen-subset 'test'\
        --batch-size 1 --beam ${beam}\
        --remove-bpe\
        --tokenizer moses\
        --source-lang $SRC_LANG \
        --target-lang $TGT_LANG \
        --scoring sacrebleu\
        > ${RESULT_PATH}generated_${beam}.txt;
done

RESULT_PATH=${LOGS_PATH}/unnormalized/
mkdir $RESULT_PATH

for beam in 5 10 20 40 60 80 100 
do
    echo ${beam}
    fairseq-generate ${BIN_DIR} \
        --task translation\
        --path ${CHECKPOINT} \
        --gen-subset 'test'\
        --batch-size 8 --beam ${beam}\
        --remove-bpe\
        --tokenizer moses\
        --source-lang $SRC_LANG \
        --target-lang $TGT_LANG \
        --scoring sacrebleu\
        --unnormalized\
        > ${RESULT_PATH}generated_${beam}.txt;
done

for beam in 400 800
do
    echo ${beam}
    fairseq-generate ${BIN_DIR} \
        --task translation\
        --path ${CHECKPOINT} \
        --gen-subset 'test'\
        --batch-size 1 --beam ${beam}\
        --remove-bpe\
        --tokenizer moses\
        --source-lang $SRC_LANG \
        --target-lang $TGT_LANG \
        --scoring sacrebleu\
        --unnormalized\
        > ${RESULT_PATH}generated_${beam}.txt;
done
```


### Reference

```
@inproceedings{provilkov-malinin-2021-multi,
    title = "Multi-Sentence Resampling: A Simple Approach to Alleviate Dataset Length Bias and Beam-Search Degradation",
    author = "Provilkov, Ivan  and
      Malinin, Andrey",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.677",
    pages = "8612--8621",
    abstract = "Neural Machine Translation (NMT) is known to suffer from a beam-search problem: after a certain point, increasing beam size causes an overall drop in translation quality. This effect is especially pronounced for long sentences. While much work was done analyzing this phenomenon, primarily for autoregressive NMT models, there is still no consensus on its underlying cause. In this work, we analyze errors that cause major quality degradation with large beams in NMT and Automatic Speech Recognition (ASR). We show that a factor that strongly contributes to the quality degradation with large beams is dataset length-bias - NMT datasets are strongly biased towards short sentences. To mitigate this issue, we propose a new data augmentation technique {--} Multi-Sentence Resampling (MSR). This technique extends the training examples by concatenating several sentences from the original dataset to make a long training example. We demonstrate that MSR significantly reduces degradation with growing beam size and improves final translation quality on the IWSTL15 En-Vi, IWSTL17 En-Fr, and WMT14 En-De datasets.",
}

```
