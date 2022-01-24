# Experiments on ImageNet data

This folder contains the code to reproduce the image classification experiments in the paper. The three steps to
reproduce them are:

1. Training the ensemble of ResNet-50 models on ImageNet;
2. Training the distilled models (Dirichlet Proxy and baselines);
3. Computing the metrics of trained models.

## Environment

To install the necessary libraries, please see [requirements.txt](./requirements.txt) for specific versions of code
dependencies.

## Data

* To obtain ImageNet, please refer to the [original website](https://image-net.org/download.php);
* For ImageNet-C, see https://github.com/hendrycks/robustness;
* Instructions to get ImageNet-A and ImageNet-O can be found at https://github.com/hendrycks/natural-adv-examples;
* ImageNet-R was taken from https://github.com/hendrycks/imagenet-r.

For the remainder of the guide, it is assumed that the original ImageNet is available via the `$INPUT_PATH/imagenet_1k`
path and all out-of-distribution datasets are in `$INPUT_PATH/imagenet_ood`. For example, all subsets of ImageNet-C are
expected to be in folders with the structure `$INPUT_PATH/imagenet_ood/imagenet-c-{distortion}-{severity}` and
ImageNet-R is in `$INPUT_PATH/imagenet_ood/imagenet-r`). For more details about data loading, please
see [data_utils.py](./data_utils.py).

## Training the baseline models

The following script trains a single ResNet-50 model on the ImageNet dataset (assumes 8 V100 GPUs):

```
python3 train.py \
  --gpus 8 \
  --accelerator ddp \
  --batch_size 256 \
  --seed 0 \
  --lr 0.1 --wd 1e-4 \
  --max_epochs 120 \
  --precision 16 \
  --weights_save_path "$SNAPSHOT_PATH" \
  --data-dir "$DATA_PATH"/imagenet_1k \
  --benchmark \
  --version base \
  --progress_bar_refresh_rate 500 \
  --logdir "$LOGS_PATH"
```

Here, `SNAPSHOT_PATH` is a variable pointing to the directory with snapshots and `LOGS_PATH` denotes the location of
TensorBoard logs written during training. This script trains a model with a random seed of 0; to obtain the ensemble,
you need to run this script with values of `--seed` from 0 to 9.

You can also download the ensemble used for our experiments: [link](https://storage.yandexcloud.net/yandex-research/proxy-dirichlet-distillation/imagenet/ensemble.tar.gz).

<details>
<summary>Training MIMO (Havasi et al., 2021) as a baseline</summary>
You can also train a multi-input multi-output (MIMO) model from the corresponding paper [1] by running the following command:

```
python3 train_mimo.py \
    --gpus 8 \
    --accelerator ddp \
    --batch_size 512 \
    --seed 0 \
    --lr 0.1 --wd 1e-4 \
    --max_epochs 150 \
    --precision 16 \
    --weights_save_path "$SNAPSHOT_PATH" \
    --benchmark \
    --version base \
    --progress_bar_refresh_rate 500 \
    --logdir "$LOGS_PATH" \
    --data-dir "$DATA_PATH"/imagenet_1k \
```

[1] [Training independent subnetworks for robust prediction](https://openreview.net/forum?id=OGg9XnKxFAH). Marton
Havasi, Rodolphe Jenatton, Stanislav Fort, Jeremiah Zhe Liu, Jasper Snoek, Balaji Lakshminarayanan, Andrew Mingbo Dai,
Dustin Tran. ICLR 2021
</details>

## Distilling the ensemble

After the ensemble is trained, place the last checkpoints from all runs into `ENSEMBLE_CHECKPOINTS_PATH`. Then, you can
train a model with Dirichlet Proxy distillation by running

```
python3 train_distillation.py \
    --gpus 8 \
    --accelerator ddp \
    --batch_size 256 \
    --seed 0 \
    --lr 0.1 --wd 1e-4 \
    --max_epochs 90 \
    --precision 16 \
    --weights_save_path "$SNAPSHOT_PATH" \
    --benchmark \
    --version base \
    --progress_bar_refresh_rate 500 \
    --logdir "$LOGS_PATH" \
    --data-dir "$DATA_PATH"/imagenet_1k \
    --loss proxy \
    --ensemble_paths "$ENSEMBLE_CHECKPOINTS_PATH"/*.ckpt \
    --model_offset 1 --target_offset 1
```

Also, you can perform standard distillation that minimizes the KL divergence between model outputs and the mean of the
ensemble:

```
python3 train_distillation.py \
    --gpus 8 \
    --accelerator ddp \
    --batch_size 256 \
    --seed 0 \
    --lr 0.1 --wd 1e-4 \
    --max_epochs 90 \
    --precision 16 \
    --weights_save_path "$SNAPSHOT_PATH" \
    --benchmark \
    --version base \
    --progress_bar_refresh_rate 500 \
    --logdir "$LOGS_PATH" \
    --data-dir "$DATA_PATH"/imagenet_1k \
    --loss forward_kl \
    --ensemble_paths "$ENSEMBLE_CHECKPOINTS_PATH"/*.ckpt
```

## Computing the metrics

* To evaluate a single model (trained either with the original objective or with distillation), you can use the
  following command:
```
python3 compute_metrics.py \
    --gpus 1 \
    --accelerator gpu \
    --batch_size 512 \
    --resume_from_checkpoint "$SNAPSHOT_PATH"/last.ckpt \
    --data-dir "$DATA_PATH"/imagenet_1k \
    -p $INPUT_PATH/imagenet_ood -j "$JSON_OUTPUT_FILE"
```
Here, `JSON_OUTPUT_FILE` is the path to the JSON file that will contain the computed metrics.

* For the ensemble metrics, use `compute_ensemble_metrics.py` for uncertainty estimates based on the original
  distributions of ensemble members or `compute_proxy_metrics.py` for the Dirichlet proxy derived from ensemble
  statistics. In each case, make sure to pass `--ensemble_paths` instead of `--resume_from_checkpoint`.

* For MIMO, use `compute_mimo_metrics.py` with the same arguments as for the single model script.