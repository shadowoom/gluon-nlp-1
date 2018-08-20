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
"""Building blocks and utility for models."""
__all__ = ['apply_weight_drop']

import functools
import warnings
import numpy as np

from mxnet import gluon, ndarray
from mxnet.gluon import Block, HybridBlock, contrib, rnn

from .parameter import WeightDropParameter


def apply_weight_drop(block, local_param_name, rate, axes=(),
                      weight_dropout_mode='training'):
    """Apply weight drop to the parameter of a block.

    Parameters
    ----------
    block : Block or HybridBlock
        The block whose parameter is to be applied weight-drop.
    local_param_name : str
        The parameter name used on the block. such as 'weight'.
    rate : float
        Fraction of the input units to drop. Must be a number between 0 and 1.
    axes : tuple of int, default ()
        The axes on which dropout mask is shared. If empty, regular dropout is applied.
    weight_drop_mode : {'training', 'always'}, default 'training'
        Whether the weight dropout should be applied only at training time, or always be applied.
    """
    if not rate:
        return

    params = block.collect_params('.*{}'.format(local_param_name))
    for full_param_name, param in params.items():
        dropped_param = WeightDropParameter(param, rate, weight_dropout_mode, axes)
        param_dicts, reg_param_dicts = _find_param(block, full_param_name, local_param_name)
        for param_dict in param_dicts:
            param_dict[full_param_name] = dropped_param
        for reg_param_dict in reg_param_dicts:
            reg_param_dict[local_param_name] = dropped_param
        local_attr = getattr(block, local_param_name)
        if local_attr == param:
            super(Block, block).__setattr__(local_param_name, dropped_param)
        else:
            if isinstance(local_attr, (list, tuple)):
                if isinstance(local_attr, tuple):
                    local_attr = list(local_attr)
                for i, v in enumerate(local_attr):
                    if v == param:
                        local_attr[i] = dropped_param
            elif isinstance(local_attr, dict):
                for k, v in local_attr:
                    if v == param:
                        local_attr[k] = dropped_param
            else:
                continue
            super(Block, block).__setattr__(local_param_name, local_attr)


def _find_param(block, full_param_name, local_param_name):
    param_dict_results = []
    reg_dict_results = []
    params = block.params

    if full_param_name in block.params._params:
        if isinstance(block, HybridBlock) and local_param_name in block._reg_params:
            reg_dict_results.append(block._reg_params)
        while params:
            if full_param_name in params._params:
                param_dict_results.append(params._params)
            if params._shared:
                params = params._shared
            else:
                break

    if block._children:
        if isinstance(block._children, list):
            children = block._children
        elif isinstance(block._children, dict):
            children = block._children.values()
        for c in children:
            pd, rd = _find_param(c, full_param_name, local_param_name)
            param_dict_results.extend(pd)
            reg_dict_results.extend(rd)

    return param_dict_results, reg_dict_results

def _get_rnn_cell(mode, num_layers, input_size, hidden_size,
                  dropout, weight_dropout,
                  var_drop_in, var_drop_state, var_drop_out):
    """create rnn cell given specs"""
    rnn_cell = rnn.SequentialRNNCell()
    with rnn_cell.name_scope():
        for i in range(num_layers):
            if mode == 'rnn_relu':
                cell = rnn.RNNCell(hidden_size, 'relu', input_size=input_size)
            elif mode == 'rnn_tanh':
                cell = rnn.RNNCell(hidden_size, 'tanh', input_size=input_size)
            elif mode == 'lstm':
                cell = rnn.LSTMCell(hidden_size, input_size=input_size)
            elif mode == 'gru':
                cell = rnn.GRUCell(hidden_size, input_size=input_size)
            if var_drop_in + var_drop_state + var_drop_out != 0:
                cell = contrib.rnn.VariationalDropoutCell(cell,
                                                          var_drop_in,
                                                          var_drop_state,
                                                          var_drop_out)

            rnn_cell.add(cell)
            if i != num_layers - 1 and dropout != 0:
                rnn_cell.add(rnn.DropoutCell(dropout))

            if weight_dropout:
                apply_weight_drop(rnn_cell, 'h2h_weight', rate=weight_dropout)

    return rnn_cell


def _get_rnn_layer(mode, num_layers, input_size, hidden_size, dropout, weight_dropout):
    """create rnn layer given specs"""
    if mode == 'rnn_relu':
        rnn_block = functools.partial(rnn.RNN, activation='relu')
    elif mode == 'rnn_tanh':
        rnn_block = functools.partial(rnn.RNN, activation='tanh')
    elif mode == 'lstm':
        rnn_block = rnn.LSTM
    elif mode == 'gru':
        rnn_block = rnn.GRU

    block = rnn_block(hidden_size, num_layers, dropout=dropout,
                      input_size=input_size)

    if weight_dropout:
        apply_weight_drop(block, 'h2h_weight', rate=weight_dropout)

    return block


def _multi_gpu_clip_global_norm_scale(arrays, max_norm):
    """Compute the global norm'scale in order to make the 2-norm smaller than `max_norm`.
    """
    assert len(arrays) > 0
    ctx = arrays[0].context
    total_norm = ndarray.add_n(*[ndarray.dot(x, x).as_in_context(ctx)
                                 for x in (arr.reshape((-1,)) for arr in arrays)])
    total_norm = ndarray.sqrt(total_norm).asscalar()
    if not np.isfinite(total_norm):
        warnings.warn(UserWarning('nan or inf is detected. Clipping results will be undefined.'),
                      stacklevel=2)
    scale = max_norm / (total_norm + 1e-8)
    return total_norm, scale


def multi_gpu_clip_global_norm(trainer, parameters, max_norm):
    """Rescales NDArrays so that the sum of their 2-norm on every context
    is smaller than `max_norm`.
    """
    trainer.allreduce_grads()
    ctx = list(parameters)[0].list_ctx()[0]
    global_grads = [p.grad(ctx) for p in parameters]
    total_norm, scale = _multi_gpu_clip_global_norm_scale(global_grads, max_norm)
    if scale < 1.0:
        for global_grad in global_grads:
            global_grad *= scale
        if len(list(parameters)[0].list_ctx()) > 1:
            for ctx in list(parameters)[0].list_ctx()[1:]:
                for p in parameters:
                    p.as_in_context(ctx)
    return total_norm
