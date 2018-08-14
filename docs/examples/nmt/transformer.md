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
from mxnet.gluon.data import ArrayDataset, SimpleDataset
from mxnet.gluon.data import DataLoader
import gluonnlp.data.batchify as btf
from gluonnlp.data import SacreMosesDetokenizer
from gluonnlp.data import ShardedDataLoader
from gluonnlp.data import ExpWidthBucket, FixedBucketSampler, WMT2014BPE
from gluonnlp.model import BeamSearchScorer
from scripts.nmt.translation import NMTModel, BeamSearchTranslator
from scripts.nmt.transformer import get_transformer_encoder_decoder
from scripts.nmt.loss import SoftmaxCEMaskedLoss, LabelSmoothing
from scripts.nmt.utils import logging_config
from scripts.nmt.bleu import _bpe_to_words, compute_bleu
import scripts.nmt._constants as _C
from scripts.nmt.dataset import TOY
```

### Set Environment

```{.python .input  n=2}
np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
ctx = mx.gpu(0)
```

### Set Hyperparameters

```{.python .input}
# parameters for dataset
demo = True
if not demo:
    dataset = 'WMT2014BPE'
else:
    dataset = 'TOY'
src_lang = 'en'
tgt_lang = 'de'
src_max_len = -1
tgt_max_len = -1

# parameters for model
num_units = 512
hidden_size = 2048
dropout = 0.1
epsilon = 0.1
num_layers = 6
num_heads = 8
scaled = True

# parameters for training
optimizer = 'adam'
epochs = 3
batch_size = 2700
test_batch_size = 256
num_accumulated = 1
lr = 2
warmup_steps = 1
save_dir = 'transformer_en_de_u512'
average_start = 1
num_buckets = 20
log_interval = 10
bleu = '13a'

#parameters for testing
beam_size = 4
lp_alpha = 0.6
lp_k = 5

logging_config(save_dir)
```

### Load and Preprocess Dataset

The following shows how to process the dataset and cache the processed dataset
for the future use. The processing steps include: 1) clip the source and target
sequences and 2) split the string input to a list of tokens and 3) map the
string token into its index in the vocabulary and 4) append EOS token to source
sentence and add BOS and EOS tokens to target sentence.

```{.python .input  n=3}
def cache_dataset(dataset, prefix):
    """Cache the processed npy dataset  the dataset into a npz

    Parameters
    ----------
    dataset : SimpleDataset
    file_path : str
    """
    if not os.path.exists(_C.CACHE_PATH):
        os.makedirs(_C.CACHE_PATH)
    src_data = np.array([ele[0] for ele in dataset])
    tgt_data = np.array([ele[1] for ele in dataset])
    np.savez(os.path.join(_C.CACHE_PATH, prefix + '.npz'), src_data=src_data, tgt_data=tgt_data)


def load_cached_dataset(prefix):
    cached_file_path = os.path.join(_C.CACHE_PATH, prefix + '.npz')
    if os.path.exists(cached_file_path):
        print('Load cached data from {}'.format(cached_file_path))
        dat = np.load(cached_file_path)
        return ArrayDataset(np.array(dat['src_data']), np.array(dat['tgt_data']))
    else:
        return None


class TrainValDataTransform(object):
    """Transform the machine translation dataset.

    Clip source and the target sentences to the maximum length. For the source sentence, append the
    EOS. For the target sentence, append BOS and EOS.

    Parameters
    ----------
    src_vocab : Vocab
    tgt_vocab : Vocab
    src_max_len : int
    tgt_max_len : int
    """

    def __init__(self, src_vocab, tgt_vocab, src_max_len, tgt_max_len):
        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab
        self._src_max_len = src_max_len
        self._tgt_max_len = tgt_max_len

    def __call__(self, src, tgt):
        if self._src_max_len > 0:
            src_sentence = self._src_vocab[src.split()[:self._src_max_len]]
        else:
            src_sentence = self._src_vocab[src.split()]
        if self._tgt_max_len > 0:
            tgt_sentence = self._tgt_vocab[tgt.split()[:self._tgt_max_len]]
        else:
            tgt_sentence = self._tgt_vocab[tgt.split()]
        src_sentence.append(self._src_vocab[self._src_vocab.eos_token])
        tgt_sentence.insert(0, self._tgt_vocab[self._tgt_vocab.bos_token])
        tgt_sentence.append(self._tgt_vocab[self._tgt_vocab.eos_token])
        src_npy = np.array(src_sentence, dtype=np.int32)
        tgt_npy = np.array(tgt_sentence, dtype=np.int32)
        return src_npy, tgt_npy


def process_dataset(dataset, src_vocab, tgt_vocab, src_max_len=-1, tgt_max_len=-1):
    start = time.time()
    dataset_processed = dataset.transform(TrainValDataTransform(src_vocab, tgt_vocab,
                                                                src_max_len,
                                                                tgt_max_len), lazy=False)
    end = time.time()
    print('Processing Time spent: {}'.format(end - start))
    return dataset_processed


def load_translation_data(dataset, src_lang='en', tgt_lang='de'):
    """Load translation dataset

    Parameters
    ----------
    dataset : str
    src_lang : str, default 'en'
    tgt_lang : str, default 'de'

    Returns
    -------

    """
    if dataset == 'WMT2014BPE':
        common_prefix = 'WMT2014BPE_{}_{}_{}_{}'.format(src_lang, tgt_lang,
                                                        src_max_len, tgt_max_len)
        data_train = WMT2014BPE('train', src_lang=src_lang, tgt_lang=tgt_lang)
        data_val = WMT2014BPE('newstest2013', src_lang=src_lang, tgt_lang=tgt_lang)
        data_test = WMT2014BPE('newstest2014', src_lang=src_lang, tgt_lang=tgt_lang, 
                               full=False)
    elif dataset == 'TOY':
        common_prefix = 'TOY_{}_{}_{}_{}'.format(src_lang, tgt_lang,
                                                 src_max_len, tgt_max_len)
        data_train = TOY('train', src_lang=src_lang, tgt_lang=tgt_lang)
        data_val = TOY('val', src_lang=src_lang, tgt_lang=tgt_lang)
        data_test = TOY('test', src_lang=src_lang, tgt_lang=tgt_lang)
    else:
        raise NotImplementedError
    src_vocab, tgt_vocab = data_train.src_vocab, data_train.tgt_vocab
    data_train_processed = load_cached_dataset(common_prefix + '_train')
    if not data_train_processed:
        data_train_processed = process_dataset(data_train, src_vocab, tgt_vocab,
                                               src_max_len, tgt_max_len)
        cache_dataset(data_train_processed, common_prefix + '_train')
    data_val_processed = load_cached_dataset(common_prefix + '_val')
    if not data_val_processed:
        data_val_processed = process_dataset(data_val, src_vocab, tgt_vocab)
        cache_dataset(data_val_processed, common_prefix + '_val')
    data_test_processed = load_cached_dataset(common_prefix + '_' + str(False) + '_test')
    if not data_test_processed:
        data_test_processed = process_dataset(data_test, src_vocab, tgt_vocab)
        cache_dataset(data_test_processed, common_prefix + '_' + str(False) + '_test')
    fetch_tgt_sentence = lambda src, tgt: tgt
    if dataset == 'WMT2014BPE':
        val_text = WMT2014('newstest2013', src_lang=src_lang, tgt_lang=tgt_lang)
        test_text = WMT2014('newstest2014', src_lang=src_lang, tgt_lang=tgt_lang,
                            full=False)
    elif dataset == 'TOY':
        val_text = data_val
        test_text = data_test
    else:
        raise NotImplementedError
    val_tgt_sentences = list(val_text.transform(fetch_tgt_sentence))
    test_tgt_sentences = list(test_text.transform(fetch_tgt_sentence))
    return data_train_processed, data_val_processed, data_test_processed, val_tgt_sentences, test_tgt_sentences, src_vocab, tgt_vocab


def get_data_lengths(dataset):
    return list(dataset.transform(lambda srg, tgt: (len(srg), len(tgt))))


data_train, data_val, data_test, val_tgt_sentences, test_tgt_sentences, src_vocab, tgt_vocab = load_translation_data(dataset=dataset, src_lang=src_lang, tgt_lang=tgt_lang)
data_train_lengths = get_data_lengths(data_train)
data_val_lengths = get_data_lengths(data_val)
data_test_lengths = get_data_lengths(data_test)

with io.open(os.path.join(save_dir, 'val_gt.txt'), 'w', encoding='utf-8') as of:
    for ele in val_tgt_sentences:
        of.write(' '.join(ele) + '\n')

with io.open(os.path.join(save_dir, 'test_gt.txt'), 'w', encoding='utf-8') as of:
    for ele in test_tgt_sentences:
        of.write(' '.join(ele) + '\n')

data_train = data_train.transform(lambda src, tgt: (src, tgt, len(src), len(tgt)), lazy=False)
data_val = SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                          for i, ele in enumerate(data_val)])
data_test = SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                           for i, ele in enumerate(data_test)])
```

### Create Sampler and DataLoader

Now, we have obtained `data_train`, `data_val`, and `data_test`. The next step
is to construct sampler and DataLoader. The first step is to construct batchify
function, which pads and stacks sequences to form mini-batch.

```{.python .input  n=4}
train_batchify_fn = btf.Tuple(btf.Pad(), btf.Pad(),
                              btf.Stack(dtype='float32'), btf.Stack(dtype='float32'))
test_batchify_fn = btf.Tuple(btf.Pad(), btf.Pad(),
                             btf.Stack(dtype='float32'), btf.Stack(dtype='float32'),
                             btf.Stack())
target_val_lengths = list(map(lambda x: x[-1], data_val_lengths))
target_test_lengths = list(map(lambda x: x[-1], data_test_lengths))

```

We can then construct bucketing samplers, which generate batches by grouping
sequences with similar lengths.

```{.python .input  n=5}
bucket_scheme = ExpWidthBucket(bucket_len_step=1.2)
train_batch_sampler = FixedBucketSampler(lengths=data_train_lengths,
                                             batch_size=batch_size,
                                             num_buckets=num_buckets,
                                             ratio=0,
                                             shuffle=True,
                                             use_average_length=True,
                                             num_shards=1,
                                             bucket_scheme=bucket_scheme)
logging.info('Train Batch Sampler:\n{}'.format(train_batch_sampler.stats()))


val_batch_sampler = FixedBucketSampler(lengths=target_val_lengths,
                                       batch_size=test_batch_size,
                                       num_buckets=num_buckets,
                                       ratio=0.0,
                                       shuffle=False,
                                       use_average_length=True,
                                       bucket_scheme=bucket_scheme)
logging.info('Valid Batch Sampler:\n{}'.format(val_batch_sampler.stats()))

test_batch_sampler = FixedBucketSampler(lengths=target_test_lengths,
                                        batch_size=test_batch_size,
                                        num_buckets=num_buckets,
                                        ratio=0.0,
                                        shuffle=False,
                                        use_average_length=True,
                                        bucket_scheme=bucket_scheme)
logging.info('Test Batch Sampler:\n{}'.format(test_batch_sampler.stats()))

```

Given the samplers, we can create DataLoader, which is iterable.

```{.python .input  n=6}
train_data_loader = ShardedDataLoader(data_train,
                                      batch_sampler=train_batch_sampler,
                                      batchify_fn=train_batchify_fn,
                                      num_workers=8)
val_data_loader = DataLoader(data_val,
                             batch_sampler=val_batch_sampler,
                             batchify_fn=test_batchify_fn,
                             num_workers=8)
test_data_loader = DataLoader(data_test,
                              batch_sampler=test_batch_sampler,
                              batchify_fn=test_batchify_fn,
                              num_workers=8)
```

## Define Transformer Model

After obtaining DataLoader, we then start to define the Transformer. The encoder and decoder of the Transformer
can be easily obtained by calling `get_transformer_encoder_decoder` function. Then, we
use the encoder and decoder in `NMTModel` to construct the Transformer model.
`model.hybridize` allows computation to be done using symbolic backend. The model architecture is shown as below:

<div style="width: 500px;">![transformer](transformer.png)</div>

```{.python .input  n=7}
encoder, decoder = get_transformer_encoder_decoder(units=num_units,
                                                   hidden_size=hidden_size,
                                                   dropout=dropout,
                                                   num_layers=num_layers,
                                                   num_heads=num_heads,
                                                   max_src_length=max(src_max_len, 500),
                                                   max_tgt_length=max(tgt_max_len, 500),
                                                   scaled=scaled)
model = NMTModel(src_vocab=src_vocab, tgt_vocab=tgt_vocab, encoder=encoder, decoder=decoder,
                 share_embed=True, embed_size=num_units, tie_weights=True,
                 embed_initializer=None, prefix='transformer_')
model.initialize(init=mx.init.Xavier(magnitude=3.0), ctx=ctx)
static_alloc = True
model.hybridize(static_alloc=static_alloc)
logging.info(model)

label_smoothing = LabelSmoothing(epsilon=epsilon, units=len(tgt_vocab))
label_smoothing.hybridize(static_alloc=static_alloc)

loss_function = SoftmaxCEMaskedLoss(sparse_label=False)
loss_function.hybridize(static_alloc=static_alloc)

test_loss_function = SoftmaxCEMaskedLoss()
test_loss_function.hybridize(static_alloc=static_alloc)

detokenizer = SacreMosesDetokenizer()
```

Here, we build the translator using the beam search

```{.python .input  n=8}
translator = BeamSearchTranslator(model=model, beam_size=beam_size,
                                  scorer=BeamSearchScorer(alpha=lp_alpha,
                                                          K=lp_k),
                                  max_length=200)
logging.info('Use beam_size={}, alpha={}, K={}'.format(beam_size, lp_alpha, lp_k))
```

We define evaluation function as follows. The `evaluate` function use beam
search translator to generate outputs for the validation and testing datasets.

```{.python .input  n=9}
def evaluate(data_loader, context=ctx):
    """Evaluate given the data loader

    Parameters
    ----------
    data_loader : DataLoader

    Returns
    -------
    avg_loss : float
        Average loss
    real_translation_out : list of list of str
        The translation output
    """
    translation_out = []
    all_inst_ids = []
    avg_loss_denom = 0
    avg_loss = 0.0
    for _, (src_seq, tgt_seq, src_valid_length, tgt_valid_length, inst_ids) \
            in enumerate(data_loader):
        src_seq = src_seq.as_in_context(context)
        tgt_seq = tgt_seq.as_in_context(context)
        src_valid_length = src_valid_length.as_in_context(context)
        tgt_valid_length = tgt_valid_length.as_in_context(context)
        # Calculating Loss
        out, _ = model(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
        loss = test_loss_function(out, tgt_seq[:, 1:], tgt_valid_length - 1).mean().asscalar()
        all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
        avg_loss += loss * (tgt_seq.shape[1] - 1)
        avg_loss_denom += (tgt_seq.shape[1] - 1)
        # Translate
        samples, _, sample_valid_length = \
            translator.translate(src_seq=src_seq, src_valid_length=src_valid_length)
        max_score_sample = samples[:, 0, :].asnumpy()
        sample_valid_length = sample_valid_length[:, 0].asnumpy()
        for i in range(max_score_sample.shape[0]):
            translation_out.append(
                [tgt_vocab.idx_to_token[ele] for ele in
                 max_score_sample[i][1:(sample_valid_length[i] - 1)]])
    avg_loss = avg_loss / avg_loss_denom
    real_translation_out = [None for _ in range(len(all_inst_ids))]
    for ind, sentence in zip(all_inst_ids, translation_out):
        real_translation_out[ind] = detokenizer(_bpe_to_words(sentence),
                                                return_str=True)
    return avg_loss, real_translation_out


def write_sentences(sentences, file_path):
    with io.open(file_path, 'w', encoding='utf-8') as of:
        for sent in sentences:
            if isinstance(sent, (list, tuple)):
                of.write(' '.join(sent) + '\n')
            else:
                of.write(sent + '\n')
```

## Training Loop

Before conducting training, we need to create trainer for updating the
parameter. In the following example, we create a trainer that uses ADAM
optimzier.

```{.python .input  n=10}
trainer = gluon.Trainer(model.collect_params(), optimizer,
                        {'learning_rate': lr, 'beta2': 0.98, 'epsilon': 1e-9})
```

We can then write the training loop. During the training, we perform the evaluation on validation and testing dataset every epoch, and record the parameters that give the hightest BLEU score on validation dataset. Before performing forward and backward, we first use `as_in_context` function to copy the mini-batch to GPU. The statement `with mx.autograd.record()` will locate Gluon backend to compute the gradients for the part inside the block. For ease of observing the convergence of the update of the `Loss` in a quick fashion, we set the `epochs = 3`. Notice that, in order to obtain the best BLEU score, we will need more epochs and large warmup steps following the original paper.

```{.python .input  n=11}
bpe = False
split_compound_word = False
tokenized = False
best_valid_bleu = 0.0
step_num = 0
warmup_steps = warmup_steps
grad_interval = num_accumulated
model.collect_params().setattr('grad_req', 'add')
average_start = (len(train_data_loader) // grad_interval) * (epochs - average_start)
average_param_dict = None
model.collect_params().zero_grad()
for epoch_id in range(epochs):
    log_avg_loss = 0
    log_wc = 0
    loss_denom = 0
    step_loss = 0
    log_start_time = time.time()
    for batch_id, seqs in enumerate(train_data_loader):
        if batch_id % grad_interval == 0:
            step_num += 1
            new_lr = lr / math.sqrt(num_units) * min(1. / math.sqrt(step_num), step_num * warmup_steps ** (-1.5))
            trainer.set_learning_rate(new_lr)
        src_wc, tgt_wc, bs = np.sum([(shard[2].sum(), shard[3].sum(), shard[0].shape[0])
                                     for shard in seqs], axis=0)
        src_wc = src_wc.asscalar()
        tgt_wc = tgt_wc.asscalar()
        loss_denom += tgt_wc - bs
        seqs = [[seq.as_in_context(context) for seq in shard]
                for context, shard in zip([ctx], seqs)]
        Ls = []
        with mx.autograd.record():
            for src_seq, tgt_seq, src_valid_length, tgt_valid_length in seqs:
                out, _ = model(src_seq, tgt_seq[:, :-1],
                               src_valid_length, tgt_valid_length - 1)
                smoothed_label = label_smoothing(tgt_seq[:, 1:])
                ls = loss_function(out, smoothed_label, tgt_valid_length - 1).sum()
                Ls.append((ls * (tgt_seq.shape[1] - 1)) / batch_size / 100.0)
        for L in Ls:
            L.backward()
        if batch_id % grad_interval == grad_interval - 1 or\
                batch_id == len(train_data_loader) - 1:
            if average_param_dict is None:
                average_param_dict = {k: v.data(ctx).copy() for k, v in
                                      model.collect_params().items()}
            trainer.step(float(loss_denom) / batch_size / 100.0)
            param_dict = model.collect_params()
            param_dict.zero_grad()
            if step_num > average_start:
                alpha = 1. / max(1, step_num - average_start)
                for name, average_param in average_param_dict.items():
                    average_param[:] += alpha * (param_dict[name].data(ctx) - average_param)
        step_loss += sum([L.asscalar() for L in Ls])
        if batch_id % grad_interval == grad_interval - 1 or\
                batch_id == len(train_data_loader) - 1:
            log_avg_loss += step_loss / loss_denom * batch_size * 100.0
            loss_denom = 0
            step_loss = 0
        log_wc += src_wc + tgt_wc
        if (batch_id + 1) % (log_interval * grad_interval) == 0:
            wps = log_wc / (time.time() - log_start_time)
            logging.info('[Epoch {} Batch {}/{}] loss={:.4f}, ppl={:.4f}, '
                         'throughput={:.2f}K wps, wc={:.2f}K'
                         .format(epoch_id, batch_id + 1, len(train_data_loader),
                                 log_avg_loss / log_interval,
                                 np.exp(log_avg_loss / log_interval),
                                 wps / 1000, log_wc / 1000))
            log_start_time = time.time()
            log_avg_loss = 0
            log_wc = 0
    mx.nd.waitall()
    valid_loss, valid_translation_out = evaluate(val_data_loader, ctx)
    valid_bleu_score, _, _, _, _ = compute_bleu([val_tgt_sentences], valid_translation_out,
                                                tokenized=tokenized, tokenizer=bleu,
                                                split_compound_word=split_compound_word,
                                                bpe=bpe)
    logging.info('[Epoch {}] valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}'
                 .format(epoch_id, valid_loss, np.exp(valid_loss), valid_bleu_score * 100))
    test_loss, test_translation_out = evaluate(test_data_loader, ctx)
    test_bleu_score, _, _, _, _ = compute_bleu([test_tgt_sentences], test_translation_out,
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
if average_start > 0:
    for k, v in model.collect_params().items():
        v.set_data(average_param_dict[k])
else:
    model.load_params(os.path.join(save_dir, 'valid_best.params'), ctx)
valid_loss, valid_translation_out = evaluate(val_data_loader, ctx)
valid_bleu_score, _, _, _, _ = compute_bleu([val_tgt_sentences], valid_translation_out,
                                            tokenized=tokenized, tokenizer=bleu, bpe=bpe,
                                            split_compound_word=split_compound_word)
logging.info('Best model valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}'
             .format(valid_loss, np.exp(valid_loss), valid_bleu_score * 100))
test_loss, test_translation_out = evaluate(test_data_loader, ctx)
test_bleu_score, _, _, _, _ = compute_bleu([test_tgt_sentences], test_translation_out,
                                           tokenized=tokenized, tokenizer=bleu, bpe=bpe,
                                           split_compound_word=split_compound_word)
logging.info('Best model test Loss={:.4f}, test ppl={:.4f}, test bleu={:.2f}'
             .format(test_loss, np.exp(test_loss), test_bleu_score * 100))
write_sentences(valid_translation_out,
                os.path.join(save_dir, 'best_valid_out.txt'))
write_sentences(test_translation_out,
                os.path.join(save_dir, 'best_test_out.txt'))
```

## Load Pretrained SOTA Transformer

Next, we will load the pretrained SOTA Transformer using the model API in GluonNLP. In this way, we can easily get access to the SOTA machine translation model and use it in your own application.

```{.python .input}
# model_name = 'transformer.average.params'
# transformer_model, src_vocab, tgt_vocab = nlp.model.get_model(model_name, src_vocab=src_vocab, tgt_vocab=tgt_vocab, encoder=encoder, decoder=decoder,
#                  share_embed=True, embed_size=num_units, tie_weights=True, prefix='transformer_')

# print(transformer_model)
# print(src_vocab)
# print(tgt_vocab)
```

Next, we will generate the SOTA results on validation and test datasets respectively.

```{.python .input}
# valid_loss, valid_translation_out = evaluate(val_data_loader, ctx)
# test_loss, test_translation_out = evaluate(test_data_loader, ctx)
# test_bleu_score, _, _, _, _ = compute_bleu([test_tgt_sentences], test_translation_out,
#                                                tokenized=tokenized, tokenizer=bleu,
#                                                split_compound_word=split_compound_word,
#                                                bpe=bpe)
```

## Conclusion

- Showcase with Transformer, we are able to support the deep neural networks for seq2seq task. We have already achieved SOTA results on the WMT 2014 English-German task.
- Gluon NLP Toolkit provides high-level APIs that could drastically simplify the development process of modeling for NLP tasks sharing the encoder-decoder structure.
- Low-level APIs in NLP Toolkit enables easy customization.

Documentation can be found at http://gluon-nlp.mxnet.io/index.html

Code is here https://github.com/dmlc/gluon-nlp

## References

[1] Vaswani, Ashish, et al. "Attention is all you need." Advances in Neural Information Processing Systems. 2017.
