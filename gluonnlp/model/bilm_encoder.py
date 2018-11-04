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

import numpy
import warnings
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import h5py
except ImportError:
    import h5py


class BiLMEncoder(gluon.Block):
    r"""Bidirectional LM encoder.

    We implement the encoder of the biLM proposed in the following work::

        @inproceedings{Peters:2018,
        author={Peters, Matthew E. and  Neumann, Mark and Iyyer, Mohit and Gardner, Matt and Clark,
        Christopher and Lee, Kenton and Zettlemoyer, Luke},
        title={Deep contextualized word representations},
        booktitle={Proc. of NAACL},
        year={2018}
        }

    Parameters
    ----------
    mode : str
        The type of RNN cell to use. Options are 'lstmpc', 'rnn_tanh', 'rnn_relu', 'lstm', 'gru'.
    num_layers : int
        The number of RNN cells in the encoder.
    input_size : int
        The initial input size of in the RNN cell.
    hidden_size : int
        The hidden size of the RNN cell.
    dropout : float
        The dropout rate to use for encoder output.
    skip_connection : bool
        Whether to add skip connections (add RNN cell input to output)
    proj_size : int
        The projection size of each LSTMPCellWithClip cell
    cell_clip : float
        Clip cell state between [-cellclip, projclip] in LSTMPCellWithClip cell
    proj_clip : float
        Clip projection between [-projclip, projclip] in LSTMPCellWithClip cell
    """
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
            self.forward_layers = nn.Sequential()
            with self.forward_layers.name_scope():
                for layer_index in range(self._num_layers):
                    forward_layer = _get_rnn_cell_clip_residual(mode=self._mode,
                                                                num_layers=1,
                                                                input_size=lstm_input_size,
                                                                hidden_size=self._hidden_size,
                                                                dropout=0
                                                                if layer_index == num_layers - 1
                                                                else self._dropout,
                                                                skip_connection=False
                                                                if layer_index == 0
                                                                else self._skip_connection,
                                                                proj_size=self._proj_size,
                                                                cell_clip=self._cell_clip,
                                                                proj_clip=self._proj_clip)

                    self.forward_layers.add(forward_layer)
                    lstm_input_size = hidden_size

            lstm_input_size = self._input_size
            self.backward_layers = nn.Sequential()
            with self.backward_layers.name_scope():
                for layer_index in range(self._num_layers):
                    backward_layer = _get_rnn_cell_clip_residual(mode=self._mode,
                                                                 num_layers=1,
                                                                 input_size=lstm_input_size,
                                                                 hidden_size=self._hidden_size,
                                                                 dropout=0
                                                                 if layer_index == num_layers - 1
                                                                 else self._dropout,
                                                                 skip_connection=False
                                                                 if layer_index == 0
                                                                 else self._skip_connection,
                                                                 proj_size=self._proj_size,
                                                                 cell_clip=self._cell_clip,
                                                                 proj_clip=self._proj_clip)

                    self.backward_layers.add(backward_layer)
                    lstm_input_size = hidden_size

    def begin_state(self, **kwargs):
        return [forward_layer.begin_state(**kwargs)
                for _, forward_layer in enumerate(self.forward_layers)], \
               [backward_layer.begin_state(**kwargs)
                for _, backward_layer in enumerate(self.backward_layers)]

    def forward(self, inputs, states): # pylint: disable=arguments-differ
        """Defines the forward computation for cache cell. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`.

        Parameters
        ----------
        inputs : List[NDArray]
            The input data. Each has layout='TNC', including:
            inputs[0] indicates the data used in forward pass.
            inputs[1] indicates the data used in backward pass.
        states : List[NDArray]
            The states. Each has layout='TNC', including:
            states[0] indicates the states used in forward pass,
            each has shape (seq_len, batch_size, num_hidden).
            states[1] indicates the states used in backward pass,
            each has shape (seq_len, batch_size, num_hidden)

        Returns
        --------
        (outputs_forward, outputs_backward): Tuple
            Including:
            outputs_forward: The output data from forward pass,
            which has layout='TNC'.
            outputs_backward: The output data from backward pass,
            which has layout='TNC'
        (states_forward, states_backward) : Tuple
            Including:
            states_forward: The out states from forward pass,
            which has the same shape with *states[0]*.
            states_backward: The out states from backward pass,
            which has the same shape with *states[1]*.
        """
        seq_len = inputs[0].shape[0]

        if not states:
            states_forward, states_backward = self.begin_state(batch_size=inputs[0].shape[1])
        else:
            states_forward, states_backward = states

        outputs_forward = []
        outputs_backward = []

        for layer_index in range(self._num_layers):
            outputs_forward.append([])
            for token_index in range(seq_len):
                if layer_index == 0:
                    output, states_forward[layer_index] = self.forward_layers[layer_index](
                        inputs[0][token_index], states_forward[layer_index])
                else:
                    output, states_forward[layer_index] = self.forward_layers[layer_index](
                        outputs_forward[layer_index-1][token_index], states_forward[layer_index])
                outputs_forward[layer_index].append(output)

            outputs_backward.append([None] * seq_len)
            for token_index in reversed(range(seq_len)):
                if layer_index == 0:
                    output, states_backward[layer_index] = self.backward_layers[layer_index](
                        inputs[1][token_index], states_backward[layer_index])
                else:
                    output, states_backward[layer_index] = self.backward_layers[layer_index](
                        outputs_backward[layer_index-1][token_index], states_backward[layer_index])
                outputs_backward[layer_index][token_index] = output

        for layer_index in range(self._num_layers):
            outputs_forward[layer_index] = mx.nd.stack(*outputs_forward[layer_index])
            outputs_backward[layer_index] = mx.nd.stack(*outputs_backward[layer_index])

        return (outputs_forward, outputs_backward), (states_forward, states_backward)

    def load_weights(self, weight_file):
        """
        Load the pre-trained weights from the file.
        """
        requires_grad = self.requires_grad

        with h5py.File(weight_file, 'r') as fin:
            for i_layer, lstms in enumerate(
                    zip(self.forward_layers, self.backward_layers)
            ):
                for j_direction, lstm in enumerate(lstms):
                    # lstm is an instance of LSTMCellWithProjection
                    #TODO: check cell size==projection_size?
                    cell_size = lstm[0].projection_size

                    dataset = fin['RNN_%s' % j_direction]['RNN']['MultiRNNCell']['Cell%s' % i_layer
                                                                                ]['LSTMCell']

                    # tensorflow packs together both W and U matrices into one matrix,
                    # but pytorch maintains individual matrices.  In addition, tensorflow
                    # packs the gates as input, memory, forget, output but pytorch
                    # uses input, forget, memory, output.  So we need to modify the weights.
                    tf_weights = numpy.transpose(dataset['W_0'][...])
                    mx_weights = tf_weights.copy()

                    # split the W from U matrices
                    input_size = lstm[0].input_size
                    input_weights = mx_weights[:, :input_size]
                    recurrent_weights = mx_weights[:, input_size:]
                    tf_input_weights = tf_weights[:, :input_size]
                    tf_recurrent_weights = tf_weights[:, input_size:]

                    # handle the different gate order convention
                    for mx_w, tf_w in [[input_weights, tf_input_weights],
                                          [recurrent_weights, tf_recurrent_weights]]:
                        mx_w[(1 * cell_size):(2 * cell_size), :] = tf_w[(2 * cell_size):(3 * cell_size), :]
                        mx_w[(2 * cell_size):(3 * cell_size), :] = tf_w[(1 * cell_size):(2 * cell_size), :]

                    lstm[0].params.get('i2h_weight').data()[:] = mx.nd.array(input_weights)
                    lstm[0].params.get('h2h_weight').data()[:] = mx.nd.array(recurrent_weights)
                    lstm[0].params.get('i2h_weight').grad_req = requires_grad
                    lstm[0].params.get('h2h_weight').grad_req = requires_grad

                    # the bias weights
                    tf_bias = dataset['B'][...]
                    # tensorflow adds 1.0 to forget gate bias instead of modifying the
                    # parameters...
                    tf_bias[(2 * cell_size):(3 * cell_size)] += 1
                    mx_bias = tf_bias.copy()
                    mx_bias[(1 * cell_size):(2 * cell_size)
                              ] = tf_bias[(2 * cell_size):(3 * cell_size)]
                    mx_bias[(2 * cell_size):(3 * cell_size)
                              ] = tf_bias[(1 * cell_size):(2 * cell_size)]
                    lstm[0].params.get('h2h_bias').data()[:] = mx.nd.array(mx_bias)
                    lstm[0].params.get('h2h_bias').grad_req = requires_grad

                    # the projection weights
                    proj_weights = numpy.transpose(dataset['W_P_0'][...])
                    lstm[0].params.get('h2r_weight').data()[:] = mx.nd.array(proj_weights)
                    lstm[0].params.get('h2r_weight').grad_req = requires_grad
