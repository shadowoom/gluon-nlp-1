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
"""Bidirectional LM encoder."""
__all__ = ['BiLMEncoder']

import mxnet as mx

from mxnet import gluon
from mxnet.gluon import nn
from .utils import _get_rnn_cell_clip_residual


class BiLMEncoder(gluon.Block):
    def __init__(self, mode, num_layers, input_size, hidden_size, dropout,
                 skip_connection, proj_size=None, cell_clip=None, proj_clip=None, **kwargs):
        super(BiLMEncoder, self).__init__(**kwargs)

        self._mode = mode
        self._num_layers = num_layers
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._skip_connection = skip_connection
        self._proj_size = proj_size
        self._cell_clip = cell_clip
        self._proj_clip = proj_clip

        with self.name_scope():
            lstm_input_size = self._input_size
            self._forward_layers = nn.HybridSequential()
            with self._forward_layers.name_scope():
                for layer_index in range(self._num_layers):
                    forward_layer = _get_rnn_cell_clip_residual(mode=self._mode,
                                                                num_layers=1,
                                                                input_size=lstm_input_size,
                                                                hidden_size=self._hidden_size,
                                                                dropout=0 if layer_index == num_layers - 1
                                                                else self._dropout,
                                                                skip_connection=False if layer_index == 0
                                                                else self._skip_connection,
                                                                proj_size=self._proj_size,
                                                                cell_clip=self._cell_clip,
                                                                proj_clip=self._proj_clip)

                    # setattr(self, 'forward_layer_{}'.format(layer_index), forward_layer)
                    # setattr(self, 'backward_layer_{}'.format(layer_index), backward_layer)

                    #TODO: check
                    # lstm_input_size = proj_size if mode == 'lstmp' else hidden_size
                    self._forward_layers.add(forward_layer)
                    lstm_input_size = hidden_size

            lstm_input_size = self._input_size
            self._backward_layers = nn.HybridSequential()
            with self._backward_layers.name_scope():
                for layer_index in range(self._num_layers):
                    backward_layer = _get_rnn_cell_clip_residual(mode=self._mode,
                                                                 num_layers=1,
                                                                 input_size=lstm_input_size,
                                                                 hidden_size=self._hidden_size,
                                                                 dropout=0 if layer_index == num_layers - 1
                                                                 else self._dropout,
                                                                 skip_connection=False if layer_index == 0
                                                                 else self._skip_connection,
                                                                 proj_size=self._proj_size,
                                                                 cell_clip=self._cell_clip,
                                                                 proj_clip=self._proj_clip)
                    self._backward_layers.add(backward_layer)
                    lstm_input_size = hidden_size

    def begin_state(self, *args, **kwargs):
        return [forward_layer.begin_state(*args, **kwargs) for _, forward_layer in enumerate(self._forward_layers)],\
               [backward_layer.begin_state(*args, **kwargs) for _, backward_layer in enumerate(self._backward_layers)]

    def forward(self, inputs, states):
        #TODO: check seq_len
        seq_len = inputs[0].shape[0]

        if not states:
            #TODO: check batch_size
            states_forward, states_backward = self.begin_state(batch_size=inputs[0].shape[1])
        else:
            states_forward, states_backward = states

        outputs_forward = []
        outputs_backward = []

        for layer_index in range(self._num_layers):
            #TODO: check the list init
            outputs_forward.append([])
            for token_index in range(seq_len):
                if layer_index == 0:
                    output, states_forward[layer_index] = self._forward_layers[layer_index](
                        inputs[0][token_index], states_forward[layer_index])
                else:
                    output, states_forward[layer_index] = self._forward_layers[layer_index](
                        outputs_forward[layer_index-1][token_index], states_forward[layer_index])
                outputs_forward[layer_index].append(output)

            # TODO: check the list init
            outputs_backward.append([None] * seq_len)
            for token_index in reversed(range(seq_len)):
                if layer_index == 0:
                    output, states_backward[layer_index] = self._backward_layers[layer_index](
                        inputs[1][token_index], states_backward[layer_index])
                else:
                    output, states_backward[layer_index] = self._backward_layers[layer_index](
                        outputs_backward[layer_index-1][token_index], states_backward[layer_index])
                outputs_backward[layer_index][token_index] = output

        for layer_index in range(self._num_layers):
            outputs_forward[layer_index] = mx.nd.stack(*outputs_forward[layer_index])
            outputs_backward[layer_index] = mx.nd.stack(*outputs_backward[layer_index])

        return (outputs_forward, outputs_backward), (states_forward, states_backward)