# Machine translation and speech recognition experiments

This folder contains the code to reproduce the experiments on WMT17 En-De and LibriSpeech datasets from the paper. The
major steps are:

1. Training the original ensemble;
2. Getting out-of-distribution datasets;
3. Generating ensemble predictions on training and validation data (machine translation only);
4. Training the distilled models (Dirichlet Proxy and baselines);
5. Computing the metrics of trained models.

## Training the ensemble

To train the original models, please see the corresponding examples:

* Machine translation: [examples/translation](../translation). Refer to the section "WMT'14 English to German (
  Convolutional)", but replace the hyperparameters with the ones for the Transformer architecture:

```bash
--arch transformer_wmt_en_de_big --share-decoder-input-output-embed \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
--dropout 0.1 --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens 4096
```

* Speech recognition: [examples/speech_recognition](../speech_recognition). Refer to the command for `vggtransformer_2`
  .

You can also download the ensembles used in our experiments:

* Machine translation: [link](https://storage.yandexcloud.net/yandex-research/proxy-dirichlet-distillation/wmt/ensemble.tar.gz)
* Speech recognition: [link](https://storage.yandexcloud.net/yandex-research/proxy-dirichlet-distillation/librispeech/ensemble.tar.gz)

## Getting out-of-distribution datasets

For machine translation, please
use [examples/structured_uncertainty/prepare-wmt14en2de-uncertainty.sh](../structured_uncertainty/prepare-wmt14en2de-uncertainty.sh)
. After that, you can use regular `fairseq-preprocess`, but make sure to have the following correspondence between
datasets:
`test5` should be `bio-ks.*`, `test6` — `test-dede.*`, `test9` — `test-perm-ende.*`, `test12` — `librispeech-tc.*`,
`test13` — `librispeech-tp.*`, `test14` — `test.fr` as the input.

For speech recognition, download the AMI corpus from https://groups.inf.ed.ac.uk/ami/download/
and the French language subset of Common Voice from https://commonvoice.mozilla.org/en/datasets.

## Generating ensemble predictions

First, generate predictions on training data:

```bash
fairseq-generate ${INPUT_DIR}/data \
   --path ${MODEL_PATHS} --max-tokens 1024 \
   --fp16 --nbest 1 --gen-subset train > ensemble_predictions.out
```

Then prepare the data for preprocessing with

```bash
python extract_predictions.py \
   --input ensemble_predictions.out \
   --output ${OUTPUT_DIR}/ensemble_predictions \
   --srclang en \
   --tgtlang de
```

After that, preprocess the training dataset and combine it with reference outputs for train/validation/test data:

```bash
fairseq-preprocess \
   --trainpref ensemble_predictions \
   --destdir ${OUTPUT_DIR}/data_generated \
   --source-lang en \
   --target-lang de \
   --srcdict ${INPUT_DIR}/data/dict.en.txt \
   --tgtdict ${INPUT_DIR}/data/dict.de.txt \
   --workers 32 

cp -r ${INPUT_DIR}/* ${OUTPUT_DIR}

PARA_DATA=${OUTPUT_DIR}/data
GEN_DATA=${OUTPUT_DIR}/data_generated
COMB_DATA=${OUTPUT_DIR}/data_combined
mkdir -p $COMB_DATA
cd $COMB_DATA

PARA_DATA="../data"
GEN_DATA="../data_generated"

for LANG in en de; do \
    ln -rs ${PARA_DATA}/dict.$LANG.txt ${COMB_DATA}/dict.$LANG.txt; \
    for EXT in bin idx; do \
        ln -rs ${PARA_DATA}/train.en-de.$LANG.$EXT ${COMB_DATA}/train.en-de.$LANG.$EXT; \
        ln -rs ${GEN_DATA}/train.en-de.$LANG.$EXT ${COMB_DATA}/train1.en-de.$LANG.$EXT; \
        for SUBSET in valid test test5 test6 test9 test12 test14; do \
            ln -rs ${PARA_DATA}/${SUBSET}.en-de.$LANG.$EXT ${COMB_DATA}/${SUBSET}.en-de.$LANG.$EXT; \
        done;
    done; \
done
```

For validation and test data, the commands are as follows:

```bash
for SUBSET in valid test; do
    fairseq-generate ${OUTPUT_DIR}/data --path ${MODEL_PATHS} \
       --max-tokens 2048 --fp16 --nbest 1 --gen-subset ${SUBSET} \
       --num-workers 32 --sacrebleu > ensemble_predictions_${SUBSET}.out
    
    python extract_predictions.py \
      --input ensemble_predictions_${SUBSET}.out \
      --output ${SUBSET}_ens_pred \
      --srclang en \
      --tgtlang de 
      
    fairseq-preprocess \
      --testpref ${SUBSET}_ens_pred \
      --destdir ${PARA_DATA} \
      --source-lang en \
      --target-lang de \
      --srcdict ${INPUT_DIR}/data/dict.en.txt \
      --tgtdict ${INPUT_DIR}/data/dict.de.txt \
      --workers 32
      
    for LANG in en de; do
        for EXT in bin idx; do
          ln -rs ln -rs ${PARA_DATA}/${SUBSET}.en-de.$LANG.$EXT ${COMB_DATA}/${SUBSET}.en-de.$LANG.$EXT;
        done
    done
done
```

## Training

### Machine translation:

For the baseline with standard distillation:

```bash
fairseq-train \
   ${COMB_DATA} \
   --upsample-primary 0 \
   --arch transformer_wmt_en_de_big \
   --tensorboard-logdir ${LOGS_DIR} \
   --share-decoder-input-output-embed \
   --num-workers 32 \
   --optimizer adam \
   --adam-betas '(0.9, 0.98)' \
   --clip-norm 10.0 \
   --lr 5e-4 \
   --lr-scheduler inverse_sqrt \
   --warmup-updates 4000 \
   --fp16 --memory-efficient-fp16 \
   --fp16-ensemble \
   --dropout 0.2 \
   --weight-decay 0.0001 \
   --criterion forward_kl_mean_distillation \
   --target-concentration mkl \
   --task distillation \
   --ensemble-paths ${MODEL_PATHS} \
   --max-tokens 1024 \
   --update-freq 32 \
   --save-dir ${CHECKPOINT_DIR} \
   --max-update 20000 \
   --keep-last-epochs 10 \
   --valid-subset valid,test,valid_ens_pred \
   --seed 0 \
   --ddp-backend=no_c10d \
   --user-dir fairseq/examples/ensemble_distillation
```

For Dirichlet Proxy distillation:

```bash
fairseq-train \
   ${COMB_DATA} \
   --upsample-primary 0 \
   --arch transformer_wmt_en_de_big \
   --tensorboard-logdir ${LOGS_DIR} \
   --share-decoder-input-output-embed \
   --num-workers 32 \
   --optimizer adam \
   --adam-betas '(0.9, 0.98)' \
   --clip-norm 10.0 \
   --lr 5e-4 \
   --lr-scheduler inverse_sqrt \
   --warmup-updates 4000 \
   --fp16 --memory-efficient-fp16 \
   --fp16-ensemble \
   --dropout 0.2 \
   --weight-decay 0.0001 \
   --criterion rkl_dirichlet_proxy_distillation \
   --target-concentration mkl \
   --model-offset 1 \
   --target-offset 1 \
   --task distillation \
   --ensemble-paths ${MODEL_PATHS} \
   --max-tokens 1024 \
   --update-freq 32 \
   --save-dir ${CHECKPOINT_DIR} \
   --max-update 20000 \
   --keep-last-epochs 10 \
   --valid-subset valid,test,valid_ens_pred \
   --seed 0 \
   --ddp-backend=no_c10d \
   --user-dir fairseq/examples/ensemble_distillation
```

## Speech recognition

Baseline:

```bash
fairseq-train \
   ${INPUT_DIR}/librispeech \
   --arch vggtransformer_2 \
   --tensorboard-logdir ${LOGS_DIR} \
   --num-workers 8 \
   --optimizer adadelta \
   --adadelta-eps 1e-8 \
   --adadelta-rho 0.95 \
   --clip-norm 10.0 \
   --lr 0.1 \
   --criterion forward_kl_mean_distillation \
   --target-concentration mkl \
   --task asr_distillation \
   --ensemble-paths ${MODEL_PATHS} \
   --max-tokens 8000 \
   --save-dir ${CHECKPOINT_DIR} \
   --max-epoch 80 \
   --anneal-start 0 \
   --anneal-end 4000 \
   --keep-last-epochs 10 \
   --valid-subset valid,test-clean,test-other \
   --seed 0 \
   --user-dir fairseq/examples/ensemble_distillation
```

Dirichlet Proxy distillation:

```bash
fairseq-train \
   ${INPUT_DIR}/librispeech \
   --arch vggtransformer_2 \
   --tensorboard-logdir ${LOGS_DIR} \
   --num-workers 8 \
   --optimizer adam \
   --adam-betas '(0.9, 0.98)' \
   --clip-norm 1.0 \
   --lr 1e-4 \
   --weight-decay 1e-8 \
   --lr-scheduler inverse_sqrt \
   --warmup-updates 4000 \
   --criterion rkl_dirichlet_proxy_distillation \
   --target-concentration mkl \
   --model-offset 1 \
   --target-offset 1 \
   --parametrization exp \
   --task asr_distillation \
   --ensemble-paths ${MODEL_PATHS} \
   --max-tokens 8000 \
   --save-dir ${CHECKPOINT_DIR} \
   --max-epoch 80 \
   --anneal-start 0 \
   --anneal-end 4000 \
   --keep-last-epochs 30 \
   --valid-subset valid,test-clean,test-other \
   --seed 0 \
   --user-dir fairseq/examples/ensemble_distillation
```

# Inference and metrics

Use the [compute_metrics.py](./compute_metrics.py) and [compute_asr_metrics.py](./compute_asr_metrics.py) scripts
correspondingly
(the `--no-distillation` flag for machine translation allows you to get ensemble metrics as well).

The arguments of these scripts are as follows:

* The input directory `-i` should contain the folder `data` with all required validation and test examples.
* The source directory `-s` is the root directory in which `fairseq` is located.
* The output directory `-o` will store the final predictions and the JSON file with metrics (unless a different path is
  specified with `-j`).
* The checkpoint file path `-c` denotes the path to the model that needs to be evaluated. You can pass a
  colon-separated list of models to evaluate the whole ensemble.
* `--max-tokens` or `--max-sentences` specifies the batch size used for generating predictions. Note that in some
  cases (e.g., ensemble inference), you will need to pass relatively small numbers (256 and 1 correspondingly) to fit
  in the GPU memory.
* `--model-offset` represents the constant added to the output Dirichlet distribution parameters to force its
  non-sparsity. It should be set to the same amount as during training.

The output JSON file contains a list of metrics for different uncertainty measures:
`Total Uncertainty-PE`, `MKL`, `Mutual Information-PE` and `EPKL` correspond to the metrics from Table 3 in the order
in which they are presented.