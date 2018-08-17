# Transformer based Machine Translation Using GluonNLP

In this notebook, we will show how to train Transformer and evaluate the pretrained model on WMT 2014 English-German dataset using GluonNLP. We will together go through: 1) load and process dataset, 2) define the Transformer model, 3) train and evaluate the model, and 4) use the state-of-the-art pretrained Transformer model.

## Preparation 

### Load MXNet and Gluon

```{.python .input  n=1}
import warnings
warnings.filterwarnings('ignore')

import time
import random
import os
import io
import logging
import math
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
from scripts import nmt

import hyperparameters as hparams
import dataprocessor
import utils
```

```{.json .output n=1}
[
 {
  "ename": "ModuleNotFoundError",
  "evalue": "No module named 'gluonnlp'",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m--------------------------------------------\u001b[0m",
   "\u001b[0;31mModuleNotFoundError\u001b[0mTraceback (most recent call last)",
   "\u001b[0;32m<ipython-input-1-87422c72d81b>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     14\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mmxnet\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgluon\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdata\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mArrayDataset\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mSimpleDataset\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     15\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mmxnet\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgluon\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdata\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mDataLoader\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 16\u001b[0;31m \u001b[0;32mimport\u001b[0m \u001b[0mgluonnlp\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mnlp\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     17\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mgluonnlp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbatchify\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mbtf\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     18\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mgluonnlp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdata\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mSacreMosesDetokenizer\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'gluonnlp'"
  ]
 }
]
```

### Set Environment

```{.python .input  n=2}
np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
ctx = mx.gpu(0)
```

### Load and Preprocess Dataset

The following shows how to process the dataset and cache the processed dataset
for the future use. The processing steps include: 1) clip the source and target
sequences and 2) split the string input to a list of tokens and 3) map the
string token into its index in the vocabulary and 4) append EOS token to source
sentence and add BOS and EOS tokens to target sentence.

```{.python .input  n=4}
data_train, data_val, data_test, val_tgt_sentences, test_tgt_sentences, src_vocab, tgt_vocab = dataprocessor.load_translation_data(dataset=hparams.dataset, src_lang=hparams.src_lang, tgt_lang=hparams.tgt_lang)
data_train_lengths = dataprocessor.get_data_lengths(data_train)
data_val_lengths = dataprocessor.get_data_lengths(data_val)
data_test_lengths = dataprocessor.get_data_lengths(data_test)

with io.open(os.path.join(save_dir, 'val_gt.txt'), 'w', encoding='utf-8') as of:
    for ele in val_tgt_sentences:
        of.write(' '.join(ele) + '\n')

with io.open(os.path.join(save_dir, 'test_gt.txt'), 'w', encoding='utf-8') as of:
    for ele in test_tgt_sentences:
        of.write(' '.join(ele) + '\n')

data_train = data_train.transform(lambda src, tgt: (src, tgt, len(src), len(tgt)), lazy=False)
data_val = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                          for i, ele in enumerate(data_val)])
data_test = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                           for i, ele in enumerate(data_test)])
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Load cached data from /home/ubuntu/cgwang/code/gluon-nlp-1/scripts/nmt/cached/TOY_en_de_-1_-1_train.npz\nLoad cached data from /home/ubuntu/cgwang/code/gluon-nlp-1/scripts/nmt/cached/TOY_en_de_-1_-1_val.npz\nLoad cached data from /home/ubuntu/cgwang/code/gluon-nlp-1/scripts/nmt/cached/TOY_en_de_-1_-1_False_test.npz\n"
 }
]
```

### Create Sampler and DataLoader

Now, we have obtained `data_train`, `data_val`, and `data_test`. The next step
is to construct sampler and DataLoader. The first step is to construct batchify
function, which pads and stacks sequences to form mini-batch.

```{.python .input  n=5}
train_batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(), nlp.data.batchify.Pad(),
                              nlp.data.batchify.Stack(dtype='float32'), nlp.data.batchify.Stack(dtype='float32'))
test_batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(), nlp.data.batchify.Pad(),
                             nlp.data.batchify.Stack(dtype='float32'), nlp.data.batchify.Stack(dtype='float32'),
                             nlp.data.batchify.Stack())
target_val_lengths = list(map(lambda x: x[-1], data_val_lengths))
target_test_lengths = list(map(lambda x: x[-1], data_test_lengths))

```

We can then construct bucketing samplers, which generate batches by grouping
sequences with similar lengths.

```{.python .input  n=6}
bucket_scheme = nlp.data.ExpWidthBucket(bucket_len_step=1.2)
train_batch_sampler = nlp.data.FixedBucketSampler(lengths=data_train_lengths,
                                             batch_size=hparams.batch_size,
                                             num_buckets=hparams.num_buckets,
                                             ratio=0.0,
                                             shuffle=True,
                                             use_average_length=True,
                                             num_shards=1,
                                             bucket_scheme=bucket_scheme)
logging.info('Train Batch Sampler:\n{}'.format(train_batch_sampler.stats()))


val_batch_sampler = nlp.data.FixedBucketSampler(lengths=target_val_lengths,
                                       batch_size=hparams.test_batch_size,
                                       num_buckets=hparams.num_buckets,
                                       ratio=0.0,
                                       shuffle=False,
                                       use_average_length=True,
                                       bucket_scheme=bucket_scheme)
logging.info('Valid Batch Sampler:\n{}'.format(val_batch_sampler.stats()))

test_batch_sampler = nlp.data.FixedBucketSampler(lengths=target_test_lengths,
                                        batch_size=hparams.test_batch_size,
                                        num_buckets=hparams.num_buckets,
                                        ratio=0.0,
                                        shuffle=False,
                                        use_average_length=True,
                                        bucket_scheme=bucket_scheme)
logging.info('Test Batch Sampler:\n{}'.format(test_batch_sampler.stats()))

```

```{.json .output n=6}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "2018-08-15 20:45:51,342 - root - Train Batch Sampler:\nFixedBucketSampler:\n  sample_num=30, batch_num=14\n  key=[(10, 11), (12, 12), (13, 13), (14, 15), (15, 16), (17, 18), (18, 20), (20, 22), (22, 24), (28, 31), (32, 36), (36, 41), (41, 47), (48, 55)]\n  cnt=[1, 1, 1, 1, 2, 2, 2, 4, 1, 2, 5, 1, 4, 3]\n  batch_size=[245, 225, 207, 180, 172, 155, 139, 136, 117, 89, 81, 72, 63, 53]\n2018-08-15 20:45:51,346 - root - Valid Batch Sampler:\nFixedBucketSampler:\n  sample_num=30, batch_num=14\n  key=[10, 11, 13, 15, 16, 18, 20, 22, 24, 31, 36, 41, 47, 55]\n  cnt=[1, 1, 1, 2, 3, 3, 2, 1, 1, 4, 3, 5, 2, 1]\n  batch_size=[25, 23, 19, 17, 16, 14, 13, 12, 11, 8, 7, 6, 5, 4]\n2018-08-15 20:45:51,348 - root - Test Batch Sampler:\nFixedBucketSampler:\n  sample_num=30, batch_num=14\n  key=[10, 11, 13, 15, 16, 18, 20, 22, 24, 31, 36, 41, 47, 55]\n  cnt=[1, 1, 1, 2, 3, 3, 2, 1, 1, 4, 3, 5, 2, 1]\n  batch_size=[25, 23, 19, 17, 16, 14, 13, 12, 11, 8, 7, 6, 5, 4]\n"
 }
]
```

Given the samplers, we can create DataLoader, which is iterable.

```{.python .input  n=7}
train_data_loader = nlp.data.ShardedDataLoader(data_train,
                                      batch_sampler=train_batch_sampler,
                                      batchify_fn=train_batchify_fn,
                                      num_workers=8)
val_data_loader = gluon.data.DataLoader(data_val,
                             batch_sampler=val_batch_sampler,
                             batchify_fn=test_batchify_fn,
                             num_workers=8)
test_data_loader = gluon.data.DataLoader(data_test,
                              batch_sampler=test_batch_sampler,
                              batchify_fn=test_batchify_fn,
                              num_workers=8)
```

## Define Transformer Model

After obtaining DataLoader, we then start to define the Transformer. The encoder and decoder of the Transformer
can be easily obtained by calling `get_transformer_encoder_decoder` function. Then, we
use the encoder and decoder in `NMTModel` to construct the Transformer model.
`model.hybridize` allows computation to be done using symbolic backend. We also use `label_smoothing`. The model architecture is shown as below:

<div style="width: 500px;">![transformer](transformer.png)</div>

```{.python .input  n=8}
encoder, decoder = nmt.transformer.get_transformer_encoder_decoder(units=hparams.num_units,
                                                   hidden_size=hparams.hidden_size,
                                                   dropout=hparams.dropout,
                                                   num_layers=hparams.num_layers,
                                                   num_heads=hparams.num_heads,
                                                   max_src_length=530,
                                                   max_tgt_length=549,
                                                   scaled=hparams.scaled)
model = nmt.translation.NMTModel(src_vocab=src_vocab, tgt_vocab=tgt_vocab, encoder=encoder, decoder=decoder,
                 share_embed=True, embed_size=hparams.num_units, tie_weights=True,
                 embed_initializer=None, prefix='transformer_')
model.initialize(init=mx.init.Xavier(magnitude=3.0), ctx=ctx)
static_alloc = True
model.hybridize(static_alloc=static_alloc)
logging.info(model)

label_smoothing = nmt.loss.LabelSmoothing(epsilon=hparams.epsilon, units=len(tgt_vocab))
label_smoothing.hybridize(static_alloc=static_alloc)

loss_function = nmt.loss.SoftmaxCEMaskedLoss(sparse_label=False)
loss_function.hybridize(static_alloc=static_alloc)

test_loss_function = nmt.loss.SoftmaxCEMaskedLoss()
test_loss_function.hybridize(static_alloc=static_alloc)

detokenizer = nlp.data.SacreMosesDetokenizer()
```

```{.json .output n=8}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "2018-08-15 20:45:57,103 - root - NMTModel(\n  (encoder): TransformerEncoder(\n    (dropout_layer): Dropout(p = 0.1, axes=())\n    (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n    (transformer_cells): HybridSequential(\n      (0): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (1): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (2): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (3): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (4): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (5): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n    )\n  )\n  (decoder): TransformerDecoder(\n    (dropout_layer): Dropout(p = 0.1, axes=())\n    (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n    (transformer_cells): HybridSequential(\n      (0): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj_in): Dense(None -> 512, linear)\n        (proj_inter): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (1): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj_in): Dense(None -> 512, linear)\n        (proj_inter): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (2): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj_in): Dense(None -> 512, linear)\n        (proj_inter): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (3): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj_in): Dense(None -> 512, linear)\n        (proj_inter): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (4): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj_in): Dense(None -> 512, linear)\n        (proj_inter): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (5): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj_in): Dense(None -> 512, linear)\n        (proj_inter): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n    )\n  )\n  (src_embed): HybridSequential(\n    (0): Embedding(358 -> 512, float32)\n    (1): Dropout(p = 0.0, axes=())\n  )\n  (tgt_embed): HybridSequential(\n    (0): Embedding(358 -> 512, float32)\n    (1): Dropout(p = 0.0, axes=())\n  )\n  (tgt_proj): Dense(None -> 381, linear)\n)\n"
 }
]
```

Here, we build the translator using the beam search

```{.python .input  n=9}
translator = nmt.translation.BeamSearchTranslator(model=model, beam_size=hparams.beam_size,
                                  scorer=nlp.model.BeamSearchScorer(alpha=hparams.lp_alpha,
                                                          K=hparams.lp_k),
                                  max_length=200)
logging.info('Use beam_size={}, alpha={}, K={}'.format(hparams.beam_size, hparams.lp_alpha, hparams.lp_k))
```

```{.json .output n=9}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "2018-08-15 20:45:57,559 - root - Use beam_size=4, alpha=0.6, K=5\n"
 }
]
```

## Training Loop

Before conducting training, we need to create trainer for updating the
parameter. In the following example, we create a trainer that uses ADAM
optimzier.

```{.python .input  n=11}
trainer = gluon.Trainer(model.collect_params(), hparams.optimizer,
                        {'learning_rate': hparams.lr, 'beta2': 0.98, 'epsilon': 1e-9})
```

We can then write the training loop. During the training, we perform the evaluation on validation and testing dataset every epoch, and record the parameters that give the hightest BLEU score on validation dataset. Before performing forward and backward, we first use `as_in_context` function to copy the mini-batch to GPU. The statement `with mx.autograd.record()` will locate Gluon backend to compute the gradients for the part inside the block. For ease of observing the convergence of the update of the `Loss` in a quick fashion, we set the `epochs = 3`. Notice that, in order to obtain the best BLEU score, we will need more epochs and large warmup steps following the original paper.

```{.python .input  n=12}
bpe = False
split_compound_word = False
tokenized = False
best_valid_bleu = 0.0
step_num = 0
warmup_steps = hparams.warmup_steps
grad_interval = hparams.num_accumulated
model.collect_params().setattr('grad_req', 'add')
average_start = (len(train_data_loader) // hparams.grad_interval) * (hparams.epochs - hparams.average_start)
average_param_dict = None
model.collect_params().zero_grad()
for epoch_id in range(epochs):
    train_one_epoch(epoch_id, model)
    mx.nd.waitall()
    # We define evaluation function as follows. The `evaluate` function use beam search translator to generate outputs for the validation and testing datasets.
    valid_loss, valid_translation_out = utils.evaluate(model, val_data_loader, test_loss_function, translator, tgt_vocab, detokenizer, ctx)
    valid_bleu_score, _, _, _, _ = nmt.bleu.compute_bleu([val_tgt_sentences], valid_translation_out,
                                                tokenized=tokenized, tokenizer=bleu,
                                                split_compound_word=split_compound_word,
                                                bpe=bpe)
    logging.info('[Epoch {}] valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}'
                 .format(epoch_id, valid_loss, np.exp(valid_loss), valid_bleu_score * 100))
    test_loss, test_translation_out = utils.evaluate(model, test_data_loader, test_loss_function, translator, tgt_vocab, detokenizer, ctx)
    test_bleu_score, _, _, _, _ = nmt.bleu.compute_bleu([test_tgt_sentences], test_translation_out,
                                               tokenized=tokenized, tokenizer=bleu,
                                               split_compound_word=split_compound_word,
                                               bpe=bpe)
    logging.info('[Epoch {}] test Loss={:.4f}, test ppl={:.4f}, test bleu={:.2f}'
                 .format(epoch_id, test_loss, np.exp(test_loss), test_bleu_score * 100))
    write_sentences(valid_translation_out,
                    os.path.join(save_dir, 'epoch{:d}_valid_out.txt').format(epoch_id))
    write_sentences(test_translation_out,
                    os.path.join(save_dir, 'epoch{:d}_test_out.txt').format(epoch_id))
    if valid_bleu_score > best_valid_bleu:
        best_valid_bleu = valid_bleu_score
        save_path = os.path.join(save_dir, 'valid_best.params')
        logging.info('Save best parameters to {}'.format(save_path))
        model.save_params(save_path)
    save_path = os.path.join(save_dir, 'epoch{:d}.params'.format(epoch_id))
    model.save_params(save_path)
save_path = os.path.join(save_dir, 'average.params')
mx.nd.save(save_path, average_param_dict)
for k, v in model.collect_params().items():
        v.set_data(average_param_dict[k])
# if average_start > 0:
    
# else:
#     model.load_params(os.path.join(save_dir, 'valid_best.params'), ctx)
valid_loss, valid_translation_out = utils.evaluate(model, val_data_loader, test_loss_function, translator, tgt_vocab, detokenizer, ctx)
valid_bleu_score, _, _, _, _ = nmt.bleu.compute_bleu([val_tgt_sentences], valid_translation_out,
                                            tokenized=tokenized, tokenizer=bleu, bpe=bpe,
                                            split_compound_word=split_compound_word)
logging.info('Best model valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}'
             .format(valid_loss, np.exp(valid_loss), valid_bleu_score * 100))
test_loss, test_translation_out = utils.evaluate(model, test_data_loader, test_loss_function, translator, tgt_vocab, detokenizer, ctx)
test_bleu_score, _, _, _, _ = nmt.bleu.compute_bleu([test_tgt_sentences], test_translation_out,
                                           tokenized=tokenized, tokenizer=bleu, bpe=bpe,
                                           split_compound_word=split_compound_word)
logging.info('Best model test Loss={:.4f}, test ppl={:.4f}, test bleu={:.2f}'
             .format(test_loss, np.exp(test_loss), test_bleu_score * 100))
write_sentences(valid_translation_out,
                os.path.join(save_dir, 'best_valid_out.txt'))
write_sentences(test_translation_out,
                os.path.join(save_dir, 'best_test_out.txt'))
```

```{.json .output n=12}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "2018-08-15 20:46:03,756 - root - [Epoch 0 Batch 10/14] loss=28.9083, ppl=3586784548312.6875, throughput=0.19K wps, wc=1.09K\n2018-08-15 20:49:18,270 - root - [Epoch 0] valid Loss=15.2710, valid ppl=4286389.2761, valid bleu=0.00\n2018-08-15 20:52:10,354 - root - [Epoch 0] test Loss=15.2710, test ppl=4286389.2761, test bleu=0.00\n2018-08-15 20:52:15,319 - root - [Epoch 1 Batch 10/14] loss=13.6171, ppl=820031.8395, throughput=0.28K wps, wc=1.15K\n2018-08-15 20:54:05,211 - root - [Epoch 1] valid Loss=9.5609, valid ppl=14198.3171, valid bleu=0.00\n2018-08-15 20:55:13,174 - root - [Epoch 1] test Loss=9.5609, test ppl=14198.3171, test bleu=0.00\n2018-08-15 20:55:15,641 - root - [Epoch 2 Batch 10/14] loss=9.1088, ppl=9034.4067, throughput=0.69K wps, wc=1.32K\n2018-08-15 20:56:25,899 - root - [Epoch 2] valid Loss=6.6393, valid ppl=764.5638, valid bleu=0.00\n2018-08-15 20:57:36,270 - root - [Epoch 2] test Loss=6.6393, test ppl=764.5638, test bleu=0.00\n2018-08-15 20:58:46,294 - root - Best model valid Loss=7.0692, valid ppl=1175.1757, valid bleu=0.00\n2018-08-15 20:59:57,204 - root - Best model test Loss=7.0692, test ppl=1175.1757, test bleu=0.00\n"
 }
]
```

## Load Pretrained SOTA Transformer

Next, we will load the pretrained SOTA Transformer using the model API in GluonNLP. In this way, we can easily get access to the SOTA machine translation model and use it in your own application.

```{.python .input  n=13}
model_name = 'transformer_en_de_512'

transformer_model, src_vocab, tgt_vocab = nmt.transformer.get_model(model_name, dataset_name='WMT2014', pretrained=True, ctx=ctx)

print(transformer_model)
print(src_vocab)
print(tgt_vocab)
```

```{.json .output n=13}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "NMTModel(\n  (encoder): TransformerEncoder(\n    (dropout_layer): Dropout(p = 0.1, axes=())\n    (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n    (transformer_cells): HybridSequential(\n      (0): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (1): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (2): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (3): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (4): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (5): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n    )\n  )\n  (decoder): TransformerDecoder(\n    (dropout_layer): Dropout(p = 0.1, axes=())\n    (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n    (transformer_cells): HybridSequential(\n      (0): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj_in): Dense(None -> 512, linear)\n        (proj_inter): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (1): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj_in): Dense(None -> 512, linear)\n        (proj_inter): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (2): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj_in): Dense(None -> 512, linear)\n        (proj_inter): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (3): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj_in): Dense(None -> 512, linear)\n        (proj_inter): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (4): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj_in): Dense(None -> 512, linear)\n        (proj_inter): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (5): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj_in): Dense(None -> 512, linear)\n        (proj_inter): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n    )\n  )\n  (src_embed): HybridSequential(\n    (0): Embedding(36794 -> True, float32)\n    (1): Dropout(p = 0.0, axes=())\n  )\n  (tgt_embed): HybridSequential(\n    (0): Embedding(36794 -> True, float32)\n    (1): Dropout(p = 0.0, axes=())\n  )\n  (tgt_proj): Dense(None -> 36794, linear)\n)\nVocab(size=36794, unk=\"<unk>\", reserved=\"['<eos>', '<bos>', '<eos>']\")\nVocab(size=36794, unk=\"<unk>\", reserved=\"['<eos>', '<bos>', '<eos>']\")\n"
 }
]
```

Next, we will generate the SOTA results on validation and test datasets respectively. For ease of illustration, we only show the loss on the TOY validation and test datasets. To be able to obtain the SOTA results, please set the `demo=False`, and we are able to achieve 27.09 as the BLEU score.

```{.python .input  n=14}
valid_loss, _ = utils.evaluate(val_data_loader, ctx)
test_loss, _ = utils.evaluate(test_data_loader, ctx)
print('Best validation loss %.2f'%(valid_loss))
print('Best test loss %.2f'%(test_loss))
```

```{.json .output n=14}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Best validation loss 7.07\nBest test loss 7.07\n"
 }
]
```

## Conclusion

- Showcase with Transformer, we are able to support the deep neural networks for seq2seq task. We have already achieved SOTA results on the WMT 2014 English-German task.
- Gluon NLP Toolkit provides high-level APIs that could drastically simplify the development process of modeling for NLP tasks sharing the encoder-decoder structure.
- Low-level APIs in NLP Toolkit enables easy customization.

Documentation can be found at http://gluon-nlp.mxnet.io/index.html

Code is here https://github.com/dmlc/gluon-nlp

## References

[1] Vaswani, Ashish, et al. "Attention is all you need." Advances in Neural Information Processing Systems. 2017.
