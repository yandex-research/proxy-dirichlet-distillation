# Scaling Ensemble Distribution Distillation to Many Classes with Proxy Targets

This repository contains the implementation of experiments
from ["Scaling Ensemble Distribution Distillation to Many Classes with Proxy Targets"](https://papers.nips.cc/paper/2021/hash/2f4ccb0f7a84f335affb418aee08a6df-Abstract.html) (NeurIPS 2021)
and ["Uncertainty Estimation in Autoregressive Structured Prediction"](https://openreview.net/forum?id=jN5y-zb5Q7m) (ICLR 2021).

## How to run

Instructions and code for each group of experiments are contained in separate subfolders:

* **Machine translation and speech recognition**: the code is based on [Fairseq](https://github.com/pytorch/fairseq);
  instructions are in [fairseq/examples/ensemble_distillation](./fairseq/examples/ensemble_distillation).
* **Image classification**: code and instructions are in the [imagenet](./imagenet) directory.

## Citation

If you have found our work or the code in this repository useful, you can cite the respective works as follows:

```
@inproceedings{ryabinin2021scaling,
	title        = {Scaling Ensemble Distribution Distillation to Many Classes with Proxy Targets},
	author       = {Max Ryabinin and Andrey Malinin and Mark Gales},
	year         = 2021,
	booktitle    = {Thirty-Fifth Conference on Neural Information Processing Systems},
	url          = {https://openreview.net/forum?id=7S3RMGVS5vO}
}
```

```
@inproceedings{malinin2021uncertainty,
	title        = {Uncertainty Estimation in Autoregressive Structured Prediction},
	author       = {Andrey Malinin and Mark Gales},
	year         = 2021,
	booktitle    = {International Conference on Learning Representations},
	url          = {https://openreview.net/forum?id=jN5y-zb5Q7m}
}
```