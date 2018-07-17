"""
Word Language Model
===================

This example shows how to build a word-level language model on WikiText-2 with Gluon NLP Toolkit.
By using the existing data pipeline tools and building blocks, the process is greatly simplified.

We implement the AWD LSTM language model proposed in the following work.

@article{merityRegOpt,
  title={{Regularizing and Optimizing LSTM Language Models}},
  author={Merity, Stephen and Keskar, Nitish Shirish and Socher, Richard},
  journal={ICLR},
  year={2018}
}
"""

# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import time
import math
import os
import sys
import mxnet as mx
from mxnet import gluon, autograd
import gluonnlp as nlp

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '..', '..'))


parser = argparse.ArgumentParser(description=
                                 'MXNet Autograd RNN/LSTM Language Model on Wikitext-2.')
parser.add_argument('--model', type=str, default='lstm',
                    help='type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=180,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropout_h', type=float, default=0.2,
                    help='dropout applied to hidden layer (0 = no dropout)')
parser.add_argument('--dropout_i', type=float, default=0.65,
                    help='dropout applied to input layer (0 = no dropout)')
parser.add_argument('--dropout_e', type=float, default=0.1,
                    help='dropout applied to embedding layer (0 = no dropout)')
parser.add_argument('--weight_dropout', type=float, default=0.5,
                    help='weight dropout applied to h2h weight matrix (0 = no weight dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--log-interval', type=int, default=372, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.params',
                    help='path to save the final model')
parser.add_argument('--eval_only', action='store_true',
                    help='Whether to only evaluate the trained model')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.'
                         '(using single gpu is suggested)')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--wd', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
#TODO: try alpha, beta=0
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation '
                         '(alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation '
                         '(beta = 0 means no regularization)')
parser.add_argument('--ntasgd', action='store_false',
                    help='Whether to apply ntasgd')
parser.add_argument('--test_mode', action='store_true',
                    help='Whether to run through the script with few examples')
parser.add_argument('--lr_update_interval', type=int, default=30,
                    help='lr udpate interval')
parser.add_argument('--lr_update_factor', type=float, default=0.1,
                    help='lr udpate factor')
args = parser.parse_args()

###############################################################################
# Load data
###############################################################################

context = [mx.cpu()] if args.gpus is None or args.gpus == '' else \
          [mx.gpu(int(x)) for x in args.gpus.split(',')]

assert args.batch_size % len(context) == 0, \
    'Total batch size must be multiple of the number of devices'

assert args.weight_dropout > 0 or (args.weight_dropout == 0 and args.alpha == 0), \
    'The alpha L2 regularization cannot be used with standard RNN, please set alpha to 0'

train_dataset, val_dataset, test_dataset = \
    [nlp.data.WikiText2(segment=segment,
                        skip_empty=False, bos=None, eos='<eos>')
     for segment in ['train', 'val', 'test']]

vocab = nlp.Vocab(counter=nlp.data.Counter(train_dataset[0]), padding_token=None, bos_token=None)

train_data = train_dataset.batchify(vocab, args.batch_size)
val_batch_size = 10
val_data = val_dataset.batchify(vocab, val_batch_size)
#TODO: modify back to 1
test_batch_size = 10
test_data = test_dataset.batchify(vocab, test_batch_size)

if args.test_mode:
    args.emsize = 200
    args.nhid = 200
    args.nlayers = 1
    args.epochs = 3
    train_data = train_data[0:100]
    val_data = val_data[0:100]
    test_data = test_data[0:100]

print(args)

###############################################################################
# Build the model
###############################################################################

ntokens = len(vocab)

if args.weight_dropout > 0:
    print('Use AWDRNN')
    model_eval = nlp.model.AWDRNN(args.model, len(vocab), args.emsize, args.nhid, args.nlayers,
                                  args.tied, args.dropout, args.weight_dropout,
                                  args.dropout_h, args.dropout_i, args.dropout_e)
    model = nlp.model.train.AWDRNN(args.model, len(vocab), args.emsize, args.nhid, args.nlayers,
                                   args.tied, args.dropout, args.weight_dropout,
                                   args.dropout_h, args.dropout_i, args.dropout_e)
else:
    model_eval = nlp.model.StandardRNN(args.model, len(vocab), args.emsize,
                                       args.nhid, args.nlayers, args.dropout, args.tied)
    model = nlp.model.train.StandardRNN(args.model, len(vocab), args.emsize,
                                        args.nhid, args.nlayers, args.dropout, args.tied)

model.initialize(mx.init.Xavier(), ctx=context)


if args.optimizer == 'sgd':
    trainer_params = {'learning_rate': args.lr,
                      'momentum': 0,
                      'wd': args.wd}
elif args.optimizer == 'adam':
    trainer_params = {'learning_rate': args.lr,
                      'wd': args.wd,
                      'beta1': 0,
                      'beta2': 0.999,
                      'epsilon': 1e-9}

#TODO: update_kv_store=False?
trainer = gluon.Trainer(model.collect_params(), args.optimizer, trainer_params, update_on_kvstore=False)

loss = gluon.loss.SoftmaxCrossEntropyLoss()
ar_loss = nlp.loss.ActivationRegularizationLoss(args.alpha)
tar_loss = nlp.loss.TemporalActivationRegularizationLoss(args.beta)


class JointActivationRegularizationLoss(gluon.loss.Loss):
    r"""Computes Joint Regularization Loss with standard loss.

    The activation regularization refer to
    gluonnlp.loss.ActivationRegularizationLoss.

    The temporal activation regularization refer to
    gluonnlp.loss.TemporalActivationRegularizationLoss.

    Parameters
    ----------
    loss : gluon.loss.Loss
        The standard loss
    ar_loss: gluonnlp.loss.ActivationRegularizationLoss
        The activation regularization
    tar_loss: gluonnlp.loss.TemporalActivationRegularizationLoss
        The temporal activation regularization

    Inputs:
        - **out**: NDArray
        output tensor with shape `(sequence_length, batch_size, input_size)`
          when `layout` is "TNC".
        - **target**: NDArray
        target tensor with shape `(sequence_length, batch_size, input_size)`
          when `layout` is "TNC".
        - **states**: the stack outputs from RNN,
        which consists of output from each time step (TNC).
        - **dropped_states**: the stack outputs from RNN with dropout,
        which consists of output from each time step (TNC).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """

    def __init__(self, l, ar_l, tar_l, weight=None, batch_axis=None, **kwargs):
        super(JointActivationRegularizationLoss, self).__init__(weight, batch_axis, **kwargs)
        self._loss = l
        self._ar_loss = ar_l
        self._tar_loss = tar_l

    def __repr__(self):
        s = 'JointActivationTemporalActivationRegularizationLoss'
        return s

    def hybrid_forward(self, F, out, target, states, dropped_states): # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        l = self._loss(out.reshape(-3, -1), target.reshape(-1,))
        l = l + self._ar_loss(*dropped_states)
        l = l + self._tar_loss(*states)
        return l


joint_loss = JointActivationRegularizationLoss(loss, ar_loss, tar_loss)

###############################################################################
# Training code
###############################################################################


def detach(hidden):
    """Transfer hidden states into new states, to detach them from the history.
    Parameters
    ----------
    hidden : NDArray
        The hidden states
    Returns
    ----------
    hidden: NDArray
        The detached hidden states
    """
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(h) for h in hidden]
    else:
        hidden = hidden.detach()
    return hidden


def get_batch(data_source, i, seq_len=None):
    """Get mini-batches of the dataset.

    Parameters
    ----------
    data_source : NDArray
        The dataset is evaluated on.
    i : int
        The index of the batch, starting from 0.
    seq_len : int
        The length of each sample in the batch.

    Returns
    -------
    data: NDArray
        The context
    target: NDArray
        The words to predict
    """
    seq_len = min(seq_len if seq_len else args.bptt, len(data_source) - 1 - i)
    data = data_source[i:i+seq_len]
    target = data_source[i+1:i+1+seq_len]
    return data, target


def evaluate(data_source, batch_size, params_file_name, ctx=None):
    """Evaluate the model on the dataset.

    Parameters
    ----------
    data_source : NDArray
        The dataset is evaluated on.
    batch_size : int
        The size of the mini-batch.
    params_file_name : str
        The parameter file to use to evaluate,
        e.g., val.params or args.save
    ctx : mx.cpu() or mx.gpu()
        The context of the computation.

    Returns
    -------
    loss: float
        The loss on the dataset
    """
    def torch_param_to_numpy(param):
        return param.data.numpy()

    def set_gluon_param(param, torch_param):
        value = torch_param_to_numpy(torch_param)
        param.set_data(mx.nd.array(value))


    total_L = 0.0
    ntotal = 0
    import torch
    if args.save == 'wiki1150.model.params':
        set_gluon_param(model_eval.collect_params(), torch.load(args.save))
    else:
        model_eval.load_parameters(params_file_name, context)
    hidden = model_eval.begin_state(batch_size, func=mx.nd.zeros, ctx=context[0])
    for i in range(0, len(data_source) - 1, args.bptt):
        data, target = get_batch(data_source, i)
        data = data.as_in_context(ctx)
        target = target.as_in_context(ctx)
        output, hidden = model_eval(data, hidden)
        hidden = detach(hidden)
        L = loss(output.reshape(-3, -1),
                 target.reshape(-1,))
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal


def train():
    """Training loop for awd language model.

    """
    best_val = float('Inf')
    start_train_time = time.time()
    parameters = model.collect_params()
    # parameters = model.collect_params().values()
    # model.collect_params().zero_grad()
    param_dict_avg = None
    t = 0
    avg_trigger = 0
    n = 5
    logs = []
    for epoch in range(args.epochs):
        total_L = 0.0
        start_epoch_time = time.time()
        start_log_interval_time = time.time()
        hiddens = [model.begin_state(args.batch_size//len(context),
                                     func=mx.nd.zeros, ctx=ctx) for ctx in context]
        batch_i, i = 0, 0
        while i < len(train_data) - 1 - 1:
            bptt = args.bptt if mx.nd.random.uniform().asscalar() < 0.95 else args.bptt / 2
            seq_len = max(5, int(mx.nd.random.normal(bptt, 5).asscalar()))
            lr_batch_start = trainer.learning_rate
            trainer.set_learning_rate(lr_batch_start*seq_len/args.bptt)

            data, target = get_batch(train_data, i, seq_len=seq_len)
            data_list = gluon.utils.split_and_load(data, context, batch_axis=1, even_split=True)
            target_list = gluon.utils.split_and_load(target, context, batch_axis=1, even_split=True)
            hiddens = detach(hiddens)
            Ls = []
            # L = 0
            with autograd.record():
                for j, (X, y, h) in enumerate(zip(data_list, target_list, hiddens)):
                    output, h, encoder_hs, dropped_encoder_hs = model(X, h)
                    l = joint_loss(output, y, encoder_hs, dropped_encoder_hs)
                    # L = L + l.as_in_context(context[0]) / X.size
                    Ls.append(l.as_in_context(context[0]) / X.size)
                    hiddens[j] = h
            for L in Ls:
                L.backward()
            # L.backward()

            # #Calculate the average of each parameter's gradient over all the context, and copy back to each context,
            # #this result in every context has the same averaged gradient regarding to each parameter Double check
            trainer.allreduce_grads()
            # # Do clipping over the parameters on the context
            # # TODO: make the global norm as one copy and transfer back to the contexts
            for d in data_list:
                grads = [p.grad(d.context) for p in parameters.values()]
                gluon.utils.clip_global_norm(grads, args.clip)

            # nlp.model.utils.multi_gpu_clip_global_norm(trainer, parameters.values(), args.clip)

            if args.ntasgd:
                if param_dict_avg is None:
                    param_dict_avg = {k.split(model._prefix)[1]: v.data(context[0]).copy()
                                      for k, v in parameters.items()}

            #TODO: the allreduce_grads have been computed twice
            trainer.update(1)

            if args.ntasgd:
                gamma = 1.0 / max(1, batch_i - avg_trigger + 2)
                # param_dict_batch_i = model.collect_params()
                # param_dict_batch_i.zero_grad()
                for name, param_avg in param_dict_avg.items():
                    param_avg[:] += gamma * (parameters['{}{}'.format(model._prefix, name)]
                                             .data(context[0]) - param_avg)

            total_L += sum([mx.nd.sum(L).asscalar() for L in Ls]) / len(context)
            trainer.set_learning_rate(lr_batch_start)

            if batch_i % args.log_interval == 0:
                cur_L = total_L / args.log_interval
                try:
                    print('[Epoch %d Batch %d/%d] current loss %.2f, ppl %.2f, '
                          'throughput %.2f samples/s, lr %.2f'
                          % (epoch, batch_i, len(train_data) // args.bptt, cur_L, math.exp(cur_L),
                             args.batch_size * args.log_interval
                             / (time.time() - start_log_interval_time),
                             lr_batch_start * seq_len / args.bptt))
                except OverflowError:
                    print('Current PPL is too large!')

                if args.ntasgd:
                    mx.nd.save('{}.val.params'.format(args.save), param_dict_avg)
                    val_L = evaluate(val_data, val_batch_size,
                                     '{}.val.params'.format(args.save), context[0])
                    try:
                        print('[Epoch %d Batch %d/%d] valid loss %.2f, valid ppl %.2f, '
                              'throughput %.2f samples/s, lr %.2f'
                              %(epoch, batch_i, len(train_data)//args.bptt, val_L, math.exp(val_L),
                                args.batch_size*args.log_interval/(time.time()-start_log_interval_time),
                                lr_batch_start*seq_len/args.bptt))
                    except OverflowError:
                        print('Val PPL is too large!')
                    if t > n and val_L > min(logs[:-n]):
                        for k, v in parameters.items():
                            param_dict_avg[k.split(model._prefix)[1]] = v.data(context[0]).copy()
                        avg_trigger = batch_i
                    logs.append(val_L)
                    t += 1

                total_L = 0.0
                start_log_interval_time = time.time()
            i += seq_len
            batch_i += 1

        mx.nd.waitall()

        print('[Epoch %d] throughput %.2f samples/s' % (
            epoch, (args.batch_size * len(train_data)) / (time.time() - start_epoch_time)))

        if args.ntasgd:
            mx.nd.save('{}.val.params'.format(args.save), param_dict_avg)
        else:
            model.save_params('{}.val.params'.format(args.save))
        val_L = evaluate(val_data, val_batch_size, '{}.val.params'.format(args.save), context[0])
        try:
            print('[Epoch %d] time cost %.2fs, valid loss %.2f, valid ppl %.2f' % (
                epoch, time.time() - start_epoch_time, val_L, math.exp(val_L)))
        except OverflowError:
            print('[Epoch %d] Val PPL is too large!' % epoch)

        if val_L < best_val:
            update_lr_epoch = 0
            best_val = val_L
            if args.ntasgd:
                mx.nd.save(args.save, param_dict_avg)
            else:
                model.save_params(args.save)
            test_L = evaluate(test_data, test_batch_size, args.save, context[0])
            try:
                print('[Epoch %d] test loss %.2f, test ppl %.2f'
                      % (epoch, test_L, math.exp(test_L)))
            except OverflowError:
                print('[Epoch %d] test PPL is too large!' % epoch)
        else:
            update_lr_epoch += 1
            if update_lr_epoch % args.lr_update_interval == 0 and update_lr_epoch != 0:
                lr_scale = trainer.learning_rate * args.lr_update_factor
                print('Learning rate after interval update %f' % lr_scale)
                trainer.set_learning_rate(lr_scale)
                update_lr_epoch = 0

        #TODO: add the lr scheduler trick (change to mxnet scheduler)


    print('Total training throughput %.2f samples/s'
          %((args.batch_size * len(train_data) * args.epochs) / (time.time() - start_train_time)))


if __name__ == '__main__':
    start_pipeline_time = time.time()
    if not args.eval_only:
        train()

    #TODO: reproduce pytorch result
    args.save = 'wiki1150.model.params'

    final_val_L = evaluate(val_data, val_batch_size, args.save, context[0])
    final_test_L = evaluate(test_data, test_batch_size, args.save, context[0])
    try:
        print('Best validation loss %.2f, val ppl %.2f' % (final_val_L, math.exp(final_val_L)))
    except OverflowError:
        print('Best val PPL is too large!')
    try:
        print('Best test loss %.2f, test ppl %.2f' % (final_test_L, math.exp(final_test_L)))
    except OverflowError:
        print('Best test PPL is too large!')
    print('Total time cost %.2fs'%(time.time()-start_pipeline_time))
