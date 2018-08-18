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
import nmt

import hyperparameters as hparams
import dataprocessor
import utils
```

### Set Environment

```{.python .input  n=2}
np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
ctx = mx.gpu(0)

nmt.utils.logging_config(hparams.save_dir)
```

### Load and Preprocess Dataset

The following shows how to process the dataset and cache the processed dataset
for the future use. The processing steps include: 1) clip the source and target
sequences and 2) split the string input to a list of tokens and 3) map the
string token into its index in the vocabulary and 4) append EOS token to source
sentence and add BOS and EOS tokens to target sentence.

```{.python .input  n=3}
data_train, data_val, data_test, val_tgt_sentences, test_tgt_sentences, src_vocab, tgt_vocab = dataprocessor.load_translation_data(dataset=hparams.dataset, src_lang=hparams.src_lang, tgt_lang=hparams.tgt_lang)
data_train_lengths = dataprocessor.get_data_lengths(data_train)
data_val_lengths = dataprocessor.get_data_lengths(data_val)
data_test_lengths = dataprocessor.get_data_lengths(data_test)

dataprocessor.write_evaluation_sentences(val_tgt_sentences, test_tgt_sentences)

data_train = data_train.transform(lambda src, tgt: (src, tgt, len(src), len(tgt)), lazy=False)
data_val = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                          for i, ele in enumerate(data_val)])
data_test = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                           for i, ele in enumerate(data_test)])
```

### Create Sampler and DataLoader

Now, we have obtained `data_train`, `data_val`, and `data_test`. The next step
is to construct sampler and DataLoader. The first step is to construct batchify
function, which pads and stacks sequences to form mini-batch.

```{.python .input  n=4}
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

```{.python .input  n=5}
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

Given the samplers, we can create DataLoader, which is iterable.

```{.python .input  n=6}
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

```{.python .input  n=7}
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

Here, we build the translator using the beam search

```{.python .input  n=8}
translator = nmt.translation.BeamSearchTranslator(model=model, beam_size=hparams.beam_size,
                                  scorer=nlp.model.BeamSearchScorer(alpha=hparams.lp_alpha,
                                                          K=hparams.lp_k),
                                  max_length=200)
logging.info('Use beam_size={}, alpha={}, K={}'.format(hparams.beam_size, hparams.lp_alpha, hparams.lp_k))
```

## Training Loop

Before conducting training, we need to create trainer for updating the
parameter. In the following example, we create a trainer that uses ADAM
optimzier.

```{.python .input  n=9}
trainer = gluon.Trainer(model.collect_params(), hparams.optimizer,
                        {'learning_rate': hparams.lr, 'beta2': 0.98, 'epsilon': 1e-9})
```

We can then write the training loop. During the training, we perform the evaluation on validation and testing dataset every epoch, and record the parameters that give the hightest BLEU score on validation dataset. Before performing forward and backward, we first use `as_in_context` function to copy the mini-batch to GPU. The statement `with mx.autograd.record()` will locate Gluon backend to compute the gradients for the part inside the block. For ease of observing the convergence of the update of the `Loss` in a quick fashion, we set the `epochs = 3`. Notice that, in order to obtain the best BLEU score, we will need more epochs and large warmup steps following the original paper.

```{.python .input  n=10}
bpe = False
split_compound_word = False
tokenized = False
best_valid_bleu = 0.0
step_num = 0
warmup_steps = hparams.warmup_steps
grad_interval = hparams.num_accumulated
model.collect_params().setattr('grad_req', 'add')
average_start = (len(train_data_loader) // grad_interval) * (hparams.epochs - hparams.average_start)
average_param_dict = {k: mx.nd.array([0]) for k, v in
                                      model.collect_params().items()}
update_average_param_dict = True
model.collect_params().zero_grad()
for epoch_id in range(hparams.epochs):
    utils.train_one_epoch(epoch_id, model, train_data_loader, trainer, label_smoothing, loss_function, grad_interval, average_param_dict, update_average_param_dict, step_num, ctx)
    mx.nd.waitall()
    # We define evaluation function as follows. The `evaluate` function use beam search translator to generate outputs for the validation and testing datasets.
    valid_loss, valid_translation_out = utils.evaluate(model, val_data_loader, test_loss_function, translator, tgt_vocab, detokenizer, ctx)
    valid_bleu_score, _, _, _, _ = nmt.bleu.compute_bleu([val_tgt_sentences], valid_translation_out,
                                                tokenized=tokenized, tokenizer=hparams.bleu,
                                                split_compound_word=split_compound_word,
                                                bpe=bpe)
    logging.info('[Epoch {}] valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}'
                 .format(epoch_id, valid_loss, np.exp(valid_loss), valid_bleu_score * 100))
    test_loss, test_translation_out = utils.evaluate(model, test_data_loader, test_loss_function, translator, tgt_vocab, detokenizer, ctx)
    test_bleu_score, _, _, _, _ = nmt.bleu.compute_bleu([test_tgt_sentences], test_translation_out,
                                               tokenized=tokenized, tokenizer=hparams.bleu,
                                               split_compound_word=split_compound_word,
                                               bpe=bpe)
    logging.info('[Epoch {}] test Loss={:.4f}, test ppl={:.4f}, test bleu={:.2f}'
                 .format(epoch_id, test_loss, np.exp(test_loss), test_bleu_score * 100))
    utils.write_sentences(valid_translation_out,
                    os.path.join(hparams.save_dir, 'epoch{:d}_valid_out.txt').format(epoch_id))
    utils.write_sentences(test_translation_out,
                    os.path.join(hparams.save_dir, 'epoch{:d}_test_out.txt').format(epoch_id))
    if valid_bleu_score > best_valid_bleu:
        best_valid_bleu = valid_bleu_score
        save_path = os.path.join(hparams.save_dir, 'valid_best.params')
        logging.info('Save best parameters to {}'.format(save_path))
        model.save_params(save_path)
    save_path = os.path.join(hparams.save_dir, 'epoch{:d}.params'.format(epoch_id))
    model.save_params(save_path)
save_path = os.path.join(hparams.save_dir, 'average.params')
mx.nd.save(save_path, average_param_dict)

if hparams.average_start > 0:
    for k, v in model.collect_params().items():
        v.set_data(average_param_dict[k])
else:
    model.load_params(os.path.join(save_dir, 'valid_best.params'), ctx)
valid_loss, valid_translation_out = utils.evaluate(model, val_data_loader, test_loss_function, translator, tgt_vocab, detokenizer, ctx)
valid_bleu_score, _, _, _, _ = nmt.bleu.compute_bleu([val_tgt_sentences], valid_translation_out,
                                            tokenized=tokenized, tokenizer=hparams.bleu, bpe=bpe,
                                            split_compound_word=split_compound_word)
logging.info('Best model valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}'
             .format(valid_loss, np.exp(valid_loss), valid_bleu_score * 100))
test_loss, test_translation_out = utils.evaluate(model, test_data_loader, test_loss_function, translator, tgt_vocab, detokenizer, ctx)
test_bleu_score, _, _, _, _ = nmt.bleu.compute_bleu([test_tgt_sentences], test_translation_out,
                                           tokenized=tokenized, tokenizer=hparams.bleu, bpe=bpe,
                                           split_compound_word=split_compound_word)
logging.info('Best model test Loss={:.4f}, test ppl={:.4f}, test bleu={:.2f}'
             .format(test_loss, np.exp(test_loss), test_bleu_score * 100))
utils.write_sentences(valid_translation_out,
                os.path.join(hparams.save_dir, 'best_valid_out.txt'))
utils.write_sentences(test_translation_out,
                os.path.join(hparams.save_dir, 'best_test_out.txt'))
```

## Load Pretrained SOTA Transformer

Next, we will load the pretrained SOTA Transformer using the model API in GluonNLP. In this way, we can easily get access to the SOTA machine translation model and use it in your own application.

```{.python .input  n=19}
model_name = 'transformer_en_de_512'

transformer_model, src_vocab, tgt_vocab = nmt.transformer.get_model(model_name, dataset_name='WMT2014', pretrained=True, ctx=ctx)

# print(transformer_model)
# print(src_vocab)
# print(tgt_vocab)

print('params0....')
params0 = model.collect_params()
print(params0)

print('params1....')
params1 = transformer_model.collect_params()
print(params1)
```

Next, we will generate the SOTA results on validation and test datasets respectively. For ease of illustration, we only show the loss on the TOY validation and test datasets. To be able to obtain the SOTA results, please set the `demo=False`, and we are able to achieve 27.09 as the BLEU score.

```{.python .input  n=20}
import utils
valid_loss, _ = utils.evaluate(transformer_model, val_data_loader, test_loss_function, translator, tgt_vocab, detokenizer, ctx)
test_loss, _ = utils.evaluate(transformer_model, test_data_loader, test_loss_function, translator, tgt_vocab, detokenizer, ctx)
print('Best validation loss %.2f'%(valid_loss))
print('Best test loss %.2f'%(test_loss))
```

## Conclusion

- Showcase with Transformer, we are able to support the deep neural networks for seq2seq task. We have already achieved SOTA results on the WMT 2014 English-German task.
- Gluon NLP Toolkit provides high-level APIs that could drastically simplify the development process of modeling for NLP tasks sharing the encoder-decoder structure.
- Low-level APIs in NLP Toolkit enables easy customization.

Documentation can be found at http://gluon-nlp.mxnet.io/index.html

Code is here https://github.com/dmlc/gluon-nlp

## References

[1] Vaswani, Ashish, et al. "Attention is all you need." Advances in Neural Information Processing Systems. 2017.
