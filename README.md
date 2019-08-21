# Prerequisites

Tensorflow version >= v0.7

[sklearn](http://scikit-learn.org/stable/index.html)

# Datasets

## IFTTT

We use the same crawler from [Quirk et al.](http://www.aclweb.org/anthology/P15-1085) to crawl recipes from IFTTT.com.

Processed data can be found in [here](./dataset/IFTTT/msr_data.pkl).

## Zapier

We additional provide a preprocessed dataset derived from [Zapier](https://zapier.com/) recipes crawled using [a crawler](https://github.com/miguelcb84/ewe-scrapers/blob/master/ewescrapers/spiders/zapier_spiders.py).

Processed data can be found under [this folder](./dataset/Zapier/).

# Usage

## Model architectures

The code includes the implementation of following models:

* BDLSTM+LA: in configs/model.jsonnet, set model/name to be "rnn",  model/decoder to be "LA".
* BDLSTM+A: in configs/model.jsonnet, set model/name to be "rnn",  model/decoder to be "attention".
* BDLSTM: in configs/model.jsonnet, set model/name to be "rnn",  don't set model/decoder(delete this line or set it to "").
* Dict+LA: in configs/model.jsonnet, set model/name to be "Dict",  model/decoder to be "LA".
* Dict+A: in configs/model.jsonnet, set model/name to be "Dict",  model/decoder to be "attention".
* Dict: in configs/model.jsonnet, set model/name to be "Dict",  don't set model/decoder(delete this line or set it to ""). 

## Run experiments

In the following we list some important arguments in `train.py`:
* `--dataset`: path to the preprocessed dataset.
* `--load-model`: path to the pretrained model (optional).
* `--config`: path to the file that stores the configuration of model architecture.
* `--logdir`: path to the directory that stores the models (optional).
* `--output`: name of the file that stores the prediction results (no need to specify the filename extension, the output is a pickle (.pkl) file).

```bash
python train.py --dataset dataset/IFTTT/msr_data.pkl --config configs/model.jsonnet --logdir model --output result
```

To ensemble results of several models:

```bash
python test_ensemble_probs.py --data dataset/IFTTT/msr_data.pkl --res result_0.pkl result_1.pkl ... result_N.pkl
```
