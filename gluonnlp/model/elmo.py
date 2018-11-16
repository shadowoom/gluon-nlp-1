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
import json
import logging
import os
import numpy
import mxnet as mx
import h5py

from mxnet import gluon, nd, cpu
from mxnet.gluon.model_zoo import model_store
try:
    from convolutional_encoder import ConvolutionalEncoder
    from bilm_encoder import BiLMEncoder
    from initializer.initializer import HighwayBias
except ImportError:
    from .convolutional_encoder import ConvolutionalEncoder
    from .bilm_encoder import BiLMEncoder
    from ..initializer.initializer import HighwayBias


# try:
#     import ConvolutionalEncoder, BiLMEncoder
# except ImportError:
#     from . import ConvolutionalEncoder, BiLMEncoder
# from gluonnlp.initializer import HighwayBias

__all__ = ['_ElmoBiLm', 'ELMoCharacterMapper']

def _make_bos_eos(
        character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length):
    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids


def add_sentence_boundary_token_ids(inputs, mask, sentence_begin_token, sentence_end_token):
    sequence_lengths = mask.sum(axis=1).asnumpy()
    inputs_shape = list(inputs.shape)
    new_shape = list(inputs_shape)
    new_shape[1] = inputs_shape[1] + 2
    inputs_with_boundary_tokens = nd.zeros(new_shape)
    if len(inputs_shape) == 2:
        inputs_with_boundary_tokens[:, 1:-1] = inputs
        inputs_with_boundary_tokens[:, 0] = sentence_begin_token
        for i, j in enumerate(sequence_lengths):
            inputs_with_boundary_tokens[i, j + 1] = sentence_end_token
        new_mask = inputs_with_boundary_tokens != 0
    elif len(inputs_shape) == 3:
        inputs_with_boundary_tokens[:, 1:-1, :] = inputs
        for i, j in enumerate(sequence_lengths):
            inputs_with_boundary_tokens[i, 0, :] = sentence_begin_token
            inputs_with_boundary_tokens[i, int(j + 1), :] = sentence_end_token
        new_mask = (inputs_with_boundary_tokens > 0).sum(axis=-1) > 0
    else:
        raise NotImplementedError
    return inputs_with_boundary_tokens, new_mask


class ELMoCharacterMapper:
    """
    Maps individual tokens to sequences of character ids, compatible with ELMo.
    To be consistent with previously trained models, we include it here as special of existing
    character indexers.
    """
    max_word_length = 50

    # char ids 0-255 come from utf-8 encoding bytes
    # assign 256-300 to special chars
    beginning_of_sentence_character = 256  # <begin sentence>
    end_of_sentence_character = 257  # <end sentence>
    beginning_of_word_character = 258  # <begin word>
    end_of_word_character = 259  # <end word>
    padding_character = 260 # <padding>

    beginning_of_sentence_characters = _make_bos_eos(
            beginning_of_sentence_character,
            padding_character,
            beginning_of_word_character,
            end_of_word_character,
            max_word_length
    )
    end_of_sentence_characters = _make_bos_eos(
            end_of_sentence_character,
            padding_character,
            beginning_of_word_character,
            end_of_word_character,
            max_word_length
    )

    bos_token = '<bos>'
    eos_token = '<eos>'


class _ElmoCharacterEncoder(gluon.Block):
    """
    Compute context insensitive token representation using pretrained biLM.

    This embedder has input character ids of size (batch_size, sequence_length, 50)
    and returns (batch_size, sequence_length + 2, embedding_dim), where embedding_dim
    is specified in the options file (typically 512).

    We add special entries at the beginning and end of each sequence corresponding
    to <S> and </S>, the beginning and end of sentence tokens.

    Note: this is a lower level class useful for advanced usage.  Most users should
    use ``ElmoTokenEmbedder`` or ``allennlp.modules.Elmo`` instead.

    Parameters
    ----------
    options_file : ``str``
        ELMo JSON options file
    weight_file : ``str``
        ELMo hdf5 weight file
    requires_grad: ``bool``, optional
        If True, compute gradient of ELMo parameters for fine tuning.

    The relevant section of the options file is something like:
    .. example-code::

        .. code-block:: python

            {'char_cnn': {
                'activation': 'relu',
                'embedding': {'dim': 4},
                'filters': [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
                'max_characters_per_token': 50,
                'n_characters': 262,
                'n_highway': 2
                }
            }
    """
    def __init__(self,
                 options_file,
                 weight_file,
                 requires_grad=False,
                 cnn_encoder='manual'):
        super(_ElmoCharacterEncoder, self).__init__()

        with open(options_file, 'r') as fin:
            self._options = json.load(fin)
        self._weight_file = weight_file
        self._requires_grad = requires_grad
        self._cnn_encoder = cnn_encoder

        self.output_dim = self._options['lstm']['projection_dim']

        # Cache the arrays for use in forward -- +1 due to masking.
        self._beginning_of_sentence_characters = mx.nd.array(
                numpy.array(ELMoCharacterMapper.beginning_of_sentence_characters) + 1
        )
        self._end_of_sentence_characters = mx.nd.array(
                numpy.array(ELMoCharacterMapper.end_of_sentence_characters) + 1
        )

        cnn_options = self._options['char_cnn']
        self._filters = cnn_options['filters']
        self._char_embed_dim = cnn_options['embedding']['dim']
        self._n_filters = sum(f[1] for f in self._filters)
        self._n_highway = cnn_options['n_highway']
        cnn_options = self._options['char_cnn']
        if cnn_options['activation'] == 'tanh':
            conv_layer_activation = 'tanh'
        elif cnn_options['activation'] == 'relu':
            conv_layer_activation = 'relu'
        else:
            raise NotImplementedError

        self._char_embedding_weights = nd.ones(shape=(ELMoCharacterMapper.padding_character+2, self._char_embed_dim))

        with self.name_scope():
            self._char_embedding = gluon.nn.Embedding(ELMoCharacterMapper.padding_character+2,
                                                      self._char_embed_dim)
            ngram_filter_sizes = []
            num_filters = []
            for _, (width, num) in enumerate(self._filters):
                ngram_filter_sizes.append(width)
                num_filters.append(num)
            self._convolutions = ConvolutionalEncoder(embed_size=self._char_embed_dim,
                                                                num_filters=tuple(num_filters),
                                                                ngram_filter_sizes=tuple(ngram_filter_sizes),
                                                                conv_layer_activation=conv_layer_activation,
                                                                num_highway=self._n_highway,
                                                                highway_bias=HighwayBias(
                                                                    nonlinear_transform_bias=0.0,
                                                                    transform_gate_bias=1.0),
                                                                output_size=self.output_dim)

    def get_output_dim(self):
        return self.output_dim

    def forward(self, inputs):  # pylint: disable=arguments-differ
        """
        Compute context insensitive token embeddings for ELMo representations.

        Parameters
        ----------
        inputs: ``torch.Tensor``
            Shape ``(batch_size, sequence_length, 50)`` of character ids representing the
            current batch.

        Returns
        -------
        Dict with keys:
        ``'token_embedding'``: ``torch.Tensor``
            Shape ``(batch_size, sequence_length + 2, embedding_dim)`` tensor with context
            insensitive token representations.
        ``'mask'``:  ``torch.Tensor``
            Shape ``(batch_size, sequence_length + 2)`` long tensor with sequence mask.
        """
        # Add BOS/EOS
        mask = (inputs > 0).sum(axis=-1) > 0
        character_ids_with_bos_eos, mask_with_bos_eos = add_sentence_boundary_token_ids(
                inputs,
                mask,
                self._beginning_of_sentence_characters,
                self._end_of_sentence_characters
        )

        # the character id embedding
        max_chars_per_token = self._options['char_cnn']['max_characters_per_token']
        # (batch_size * sequence_length, max_chars_per_token, embed_dim)
        character_embedding = self._char_embedding(character_ids_with_bos_eos.reshape(-1, max_chars_per_token))

        character_embedding = nd.transpose(character_embedding, axes=(1, 0, 2))
        token_embedding = self._convolutions(character_embedding)

        # reshape to (batch_size, sequence_length, embedding_dim)
        batch_size, sequence_length, _ = character_ids_with_bos_eos.shape

        return mask_with_bos_eos, token_embedding.reshape(batch_size, sequence_length, -1)

    def _load_weights(self):
        self._load_char_embedding()
        self._load_cnn_weights()
        self._load_highway()
        self._load_projection()

    def _load_char_embedding(self):
        with h5py.File(self._weight_file, 'r') as fin:
            char_embed_weights = fin['char_embed'][...]
            # print(list(fin.keys()))
            # print(fin.values())
            # print(fin)

        weights = numpy.zeros(
                (char_embed_weights.shape[0] + 1, char_embed_weights.shape[1]),
                dtype='float32'
        )
        weights[1:, :] = char_embed_weights
        self._char_embedding_weights = nd.array(weights)

    def _load_cnn_weights(self):
        if self._cnn_encoder == 'manual':
            for i, conv in enumerate(self._convolutions):
                # load the weights
                with h5py.File(self._weight_file, 'r') as fin:
                    weight = fin['CNN']['W_cnn_{}'.format(i)][...]
                    bias = fin['CNN']['b_cnn_{}'.format(i)][...]

                w_reshaped = numpy.transpose(weight.squeeze(axis=0), axes=(2, 1, 0))
                if w_reshaped.shape != tuple(conv.weight.data().shape):
                    raise ValueError("Invalid weight file")
                conv.weight.data()[:] = nd.array(w_reshaped)
                conv.bias.data()[:] = nd.array(bias)

                conv.weight.grad_req = self._requires_grad
                conv.bias.grad_req = self._requires_grad
        elif self._cnn_encoder == 'encoder':
            for i, conv_seq in enumerate(self._convolutions._convs):
                # load the weights
                conv = conv_seq[0]
                with h5py.File(self._weight_file, 'r') as fin:
                    weight = fin['CNN']['W_cnn_{}'.format(i)][...]
                    bias = fin['CNN']['b_cnn_{}'.format(i)][...]

                w_reshaped = numpy.transpose(weight.squeeze(axis=0), axes=(2, 1, 0))
                if w_reshaped.shape != tuple(conv.weight.data().shape):
                    raise ValueError("Invalid weight file")
                conv.weight.data()[:] = nd.array(w_reshaped)
                conv.bias.data()[:] = nd.array(bias)

                conv.weight.grad_req = self._requires_grad
                conv.bias.grad_req = self._requires_grad



    def _load_highway(self):
        #TODO
        # # pylint: disable=protected-access
        # # the highway layers have same dimensionality as the number of cnn filters
        # cnn_options = self._options['char_cnn']
        # filters = cnn_options['filters']
        # n_filters = sum(f[1] for f in filters)
        # n_highway = cnn_options['n_highway']
        #
        # # create the layers, and load the weights
        # self._highways = nlp.model.Highway(input_size=n_filters, num_layers=n_highway, activation='relu')
        if self._cnn_encoder == 'manual':
            for k in range(self._n_highway):
                # The AllenNLP highway is one matrix multplication with concatenation of
                # transform and carry weights.
                with h5py.File(self._weight_file, 'r') as fin:
                    # The weights are transposed due to multiplication order assumptions in tf
                    # vs pytorch (tf.matmul(X, W) vs pytorch.matmul(W, X))
                    w_transform = numpy.transpose(fin['CNN_high_{}'.format(k)]['W_transform'][...])
                    # -1.0 since AllenNLP is g * x + (1 - g) * f(x) but tf is (1 - g) * x + g * f(x)
                    w_carry = -1.0 * numpy.transpose(fin['CNN_high_{}'.format(k)]['W_carry'][...])
                    weight = numpy.concatenate([w_transform, w_carry], axis=0)
                    self._highways.hnet[k].weight.data()[:] = nd.array(weight)
                    self._highways.hnet[k].weight.grad_req = self._requires_grad

                    b_transform = fin['CNN_high_{}'.format(k)]['b_transform'][...]
                    b_carry = -1.0 * fin['CNN_high_{}'.format(k)]['b_carry'][...]
                    bias = numpy.concatenate([b_transform, b_carry], axis=0)
                    self._highways.hnet[k].bias.data()[:] = nd.array(bias)
                    self._highways.hnet[k].bias.grad_req = self._requires_grad
        elif self._cnn_encoder == 'encoder':
            for k in range(self._n_highway):
                # The AllenNLP highway is one matrix multplication with concatenation of
                # transform and carry weights.
                with h5py.File(self._weight_file, 'r') as fin:
                    # The weights are transposed due to multiplication order assumptions in tf
                    # vs pytorch (tf.matmul(X, W) vs pytorch.matmul(W, X))
                    w_transform = numpy.transpose(fin['CNN_high_{}'.format(k)]['W_transform'][...])
                    # -1.0 since AllenNLP is g * x + (1 - g) * f(x) but tf is (1 - g) * x + g * f(x)
                    w_carry = -1.0 * numpy.transpose(fin['CNN_high_{}'.format(k)]['W_carry'][...])
                    weight = numpy.concatenate([w_transform, w_carry], axis=0)
                    self._convolutions._highways.hnet[k].weight.data()[:] = nd.array(weight)
                    self._convolutions._highways.hnet[k].weight.grad_req = self._requires_grad

                    b_transform = fin['CNN_high_{}'.format(k)]['b_transform'][...]
                    b_carry = -1.0 * fin['CNN_high_{}'.format(k)]['b_carry'][...]
                    bias = numpy.concatenate([b_transform, b_carry], axis=0)
                    self._convolutions._highways.hnet[k].bias.data()[:] = nd.array(bias)
                    self._convolutions._highways.hnet[k].bias.grad_req = self._requires_grad

    def _load_projection(self):
        #TODO
        # cnn_options = self._options['char_cnn']
        # filters = cnn_options['filters']
        # n_filters = sum(f[1] for f in filters)
        #
        # self._projection = gluon.nn.Dense(in_units=n_filters, units=self.output_dim, use_bias=True)
        if self._cnn_encoder == 'manual':
            with h5py.File(self._weight_file, 'r') as fin:
                weight = fin['CNN_proj']['W_proj'][...]
                bias = fin['CNN_proj']['b_proj'][...]
                self._projection.weight.data()[:] = nd.array(numpy.transpose(weight))
                self._projection.bias.data()[:] = nd.array(bias)

                self._projection.weight.grad_req = self._requires_grad
                self._projection.bias.grad_req = self._requires_grad
        elif self._cnn_encoder == 'encoder':
            with h5py.File(self._weight_file, 'r') as fin:
                weight = fin['CNN_proj']['W_proj'][...]
                bias = fin['CNN_proj']['b_proj'][...]
                self._convolutions._projection.weight.data()[:] = nd.array(numpy.transpose(weight))
                self._convolutions._projection.bias.data()[:] = nd.array(bias)

                self._convolutions._projection.weight.grad_req = self._requires_grad
                self._convolutions._projection.bias.grad_req = self._requires_grad


class _ElmoBiLm(gluon.Block):
    """
    Run a pre-trained bidirectional language model, outputing the activations at each
    layer for weighting together into an ELMo representation (with
    ``allennlp.modules.seq2seq_encoders.Elmo``).  This is a lower level class, useful
    for advanced uses, but most users should use ``allennlp.modules.seq2seq_encoders.Elmo``
    directly.

    Parameters
    ----------
    options_file : ``str``
        ELMo JSON options file
    weight_file : ``str``
        ELMo hdf5 weight file
    requires_grad: ``bool``, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    vocab_to_cache : ``List[str]``, optional, (default = 0.5).
        A list of words to pre-compute and cache character convolutions
        for. If you use this option, _ElmoBiLm expects that you pass word
        indices of shape (batch_size, timesteps) to forward, instead
        of character indices. If you use this option and pass a word which
        wasn't pre-cached, this will break.
    """
    def __init__(self,
                 options_file,
                 weight_file=None,
                 requires_grad='null',
                 vocab_to_cache=None,
                 cnn_encoder='manual'):
        super(_ElmoBiLm, self).__init__()

        self._options_file = options_file

        self._weight_file = weight_file

        self._token_embedder = _ElmoCharacterEncoder(options_file, weight_file, requires_grad=requires_grad, cnn_encoder=cnn_encoder)

        self._requires_grad = requires_grad
        if requires_grad and vocab_to_cache:
            logging.warning("You are fine tuning ELMo and caching char CNN word vectors. "
                            "This behaviour is not guaranteed to be well defined, particularly. "
                            "if not all of your inputs will occur in the vocabulary cache.")
        # This is an embedding, used to look up cached
        # word vectors built from character level cnn embeddings.
        self._word_embedding = None
        self._bos_embedding = None
        self._eos_embedding = None

        with open(options_file, 'r') as fin:
            options = json.load(fin)
        if not options['lstm'].get('use_skip_connections'):
            raise NotImplementedError
        with self.name_scope():
            self._elmo_lstm = BiLMEncoder(mode='lstmpc',
                                                    input_size=options['lstm']['projection_dim'],
                                                    hidden_size=options['lstm']['dim'],
                                                    proj_size=options['lstm']['projection_dim'],
                                                    num_layers=options['lstm']['n_layers'],
                                                    cell_clip=options['lstm']['cell_clip'],
                                                    proj_clip=options['lstm']['proj_clip'])

    def get_output_dim(self):
        return 2 * self._token_embedder.get_output_dim()

    def forward(self,
                inputs,
                states=None,
                word_inputs=None):
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``, required.
            Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
        word_inputs : ``torch.Tensor``, required.
            If you passed a cached vocab, you can in addition pass a tensor of shape ``(batch_size, timesteps)``,
            which represent word ids which have been pre-cached.

        Returns
        -------
        Dict with keys:

        ``'activations'``: ``List[torch.Tensor]``
            A list of activations at each layer of the network, each of shape
            ``(batch_size, timesteps + 2, embedding_dim)``
        ``'mask'``:  ``torch.Tensor``
            Shape ``(batch_size, timesteps + 2)`` long tensor with sequence mask.

        Note that the output tensors all include additional special begin and end of sequence
        markers.
        """
        if self._word_embedding is not None and word_inputs is not None:
            try:
                mask_without_bos_eos = word_inputs > 0
                # The character cnn part is cached - just look it up.
                embedded_inputs = self._word_embedding(word_inputs) # type: ignore
                # shape (batch_size, timesteps + 2, embedding_dim)
                type_representation, mask = add_sentence_boundary_token_ids(
                        embedded_inputs,
                        mask_without_bos_eos,
                        self._bos_embedding,
                        self._eos_embedding
                )
            except RuntimeError:
                # Back off to running the character convolutions,
                # as we might not have the words in the cache.
                token_embedding = self._token_embedder(inputs)
                mask = token_embedding['mask']
                type_representation = token_embedding['token_embedding']
        else:
            mask, type_representation = self._token_embedder(inputs)
        lstm_outputs, states = self._elmo_lstm(type_representation, states, mask)
        lstm_outputs = lstm_outputs.transpose(axes=(0, 2, 1, 3))

        # Prepare the output.  The first layer is duplicated.
        # Because of minor differences in how masking is applied depending
        # on whether the char cnn layers are cached, we'll be defensive and
        # multiply by the mask here. It's not strictly necessary, as the
        # mask passed on is correct, but the values in the padded areas
        # of the char cnn representations can change.
        #TODO:
        # output_tensors = [
        #         nd.concat(*[type_representation, type_representation], dim=-1)
        #         * mask.expand_dims(axis=-1)
        # ]
        # for layer_activations in nd.split(data=lstm_outputs, num_outputs=lstm_outputs.shape[0], axis=0):
        #     output_tensors.append(layer_activations.squeeze(axis=0))

        # return output_tensors, mask
        # return lstm_outputs, mask
        output = [
            mx.nd.concat(*[type_representation, type_representation], dim=-1) * mask.expand_dims(axis=-1)
        ]
        for layer_activations in mx.nd.split(lstm_outputs, lstm_outputs.shape[0], axis=0):
            output.append(layer_activations.squeeze(axis=0))

        # return {
        #     'activations': output_tensors,
        #     'mask': mask,
        # }
        return output, states, mask

model_store._model_sha1.update(
    {name: checksum for checksum, name in [
        ('a416351377d837ef12d17aae27739393f59f0b82', 'standard_lstm_lm_1500_wikitext-2'),
        ('631f39040cd65b49f5c8828a0aba65606d73a9cb', 'standard_lstm_lm_650_wikitext-2'),
        ('b233c700e80fb0846c17fe14846cb7e08db3fd51', 'standard_lstm_lm_200_wikitext-2'),
        ('f9562ed05d9bcc7e1f5b7f3c81a1988019878038', 'awd_lstm_lm_1150_wikitext-2'),
        ('e952becc7580a0b5a6030aab09d0644e9a13ce18', 'awd_lstm_lm_600_wikitext-2'),
        ('6bb3e991eb4439fabfe26c129da2fe15a324e918', 'big_rnn_lm_2048_512_gbw')
    ]})

def standard_lstm_lm_200(dataset_name=None, vocab=None, pretrained=False, ctx=cpu(),
                         root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""Standard 2-layer LSTM language model with tied embedding and output weights.

    Both embedding and hidden dimensions are 200.

    Parameters
    ----------
    dataset_name : str or None, default None
        The dataset name on which the pre-trained model is trained.
        Options are 'wikitext-2'. If specified, then the returned vocabulary is extracted from
        the training set of the dataset.
        If None, then vocab is required, for specifying embedding weight size, and is directly
        returned.
        The pre-trained model achieves 108.25/102.26 ppl on Val and Test of wikitext-2 respectively.
    vocab : gluonnlp.Vocab or None, default None
        Vocabulary object to be used with the language model.
        Required when dataset_name is not specified.
    pretrained : bool, default False
        Whether to load the pre-trained weights for model.
    ctx : Context, default CPU
        The context in which to load the pre-trained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    gluon.Block, gluonnlp.Vocab
    """
    predefined_args = {'embed_size': 200,
                       'hidden_size': 200,
                       'mode': 'lstm',
                       'num_layers': 2,
                       'tie_weights': True,
                       'dropout': 0.2}
    mutable_args = ['dropout']
    assert all((k not in kwargs or k in mutable_args) for k in predefined_args), \
           'Cannot override predefined model settings.'
    predefined_args.update(kwargs)
    # return _get_rnn_model(StandardRNN, 'standard_lstm_lm_200', dataset_name, vocab, pretrained,
    #                       ctx, root, **predefined_args)


def print_outputs(outputs):
    print('0-f')
    print(outputs[0][0][0][:1, :1, :2])
    print('0-b')
    print(outputs[0][1][0][:1, :1, :2])
    print('1-f')
    print(outputs[0][0][1][:1, :1, :2])
    print('1-b')
    print(outputs[0][1][1][:1, :1, :2])

def print_outputs_nd(lstm_outputs):
    print('0-f')
    print(lstm_outputs[0][:1,:1,:2])
    print('0-b')
    print(lstm_outputs[0][:1,:1,int(lstm_outputs.shape[3]/2):int(lstm_outputs.shape[3]/2)+2])
    print('1-f')
    print(lstm_outputs[1][:1,:1,:2])
    print('1-b')
    print(lstm_outputs[1][:1,:1,int(lstm_outputs.shape[3]/2):int(lstm_outputs.shape[3]/2)+2])


data_dir = '/Users/chgwang/Documents/code/elmo-data/'
options_file = 'elmo_2x4096_512_2048cnn_2xhighway_options.json'
weight_file = 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

# print('4!!!')
# model4 = _ElmoBiLm(options_file=data_dir + options_file,
#                    weight_file=data_dir + weight_file,
#                    lstm_mode='manual',
#                    cnn_encoder='encoder')
# print(model4)
# model4.initialize()
# model4._token_embedder._load_weights()
# model4._token_embedder._char_embedding.weight.set_data(model4._token_embedder._char_embedding_weights)
# model4._elmo_lstm.load_weights(model4._weight_file, model4._requires_grad)
# model4.save_parameters(data_dir + weight_file + '.params')
# model4.hybridize()
# inputs = nd.ones(shape=(20,35,50))
# begin_state = model4._elmo_lstm.begin_state(mx.nd.zeros, batch_size=20)
# outputs4, mask4 = model4(inputs, begin_state)
# print_outputs_nd(outputs4[0])

if __name__ == '__main__':
    print('5!!!')
    model5 = _ElmoBiLm(options_file=data_dir + options_file)
    print(model5)
    model5.load_parameters(data_dir + weight_file + '.params')
    model5.hybridize()
    inputs = nd.ones(shape=(20,35,50))
    begin_state = model5._elmo_lstm.begin_state(mx.nd.zeros, batch_size=20)
    outputs5, mask5 = model5(inputs, begin_state)

# print_outputs_nd(outputs5[0])

# from numpy.testing import assert_almost_equal
# import torch
# t_output = torch.load(data_dir + weight_file + '.tout')
# # assert_almost_equal(outputs4[0].asnumpy(), t_output.numpy(), decimal=5)
# assert_almost_equal(outputs5[0].asnumpy(), t_output.numpy(), decimal=5)
# print('equal with torch!')

#
# def almost_equal(outputs0, outputs1, mask0, mask1):
#     for out0, out1 in zip(outputs0, outputs1):
#         for out00, out10 in zip(out0, out1):
#             for out000, out100 in zip(out00, out10):
#                 if isinstance(out000, mx.ndarray.ndarray.NDArray) and isinstance(out100, mx.ndarray.ndarray.NDArray):
#                     assert_almost_equal(out000.asnumpy(), out100.asnumpy(), decimal=6)
#                 else:
#                     for out0000, out1000 in zip(out000, out100):
#                         assert_almost_equal(out0000.asnumpy(), out1000.asnumpy(), decimal=6)
#     assert_almost_equal(mask0.asnumpy(), mask1.asnumpy(), decimal=6)
#
# print('0')
# almost_equal(outputs0, outputs1, mask0, mask1)
#
# print('1')
# almost_equal(outputs2, outputs3, mask2, mask3)
#
# print('2')
# almost_equal(outputs0, outputs2, mask0, mask2)
#
# print('3')
# almost_equal(outputs1, outputs3, mask1, mask3)
#
# print('4')
# almost_equal(outputs2, outputs4, mask2, mask4)
#
# print('5')
# almost_equal(outputs4, outputs5, mask4, mask5)
#
# print('6')
# almost_equal(outputs1, outputs5, mask1, mask5)


# model = nlp.model.BiLMEncoderUnroll('lstmpc', 2, input_size=3, hidden_size=5, dropout=0.0,
#                  skip_connection=True, proj_size=3, cell_clip=3, proj_clip=3)
# model.initialize(init=mx.initializer.One())
# model.hybridize()
# inputs = mx.nd.ones(shape=(1,2,3))
# outputs = model(inputs)
# print(outputs[0])

# model = nlp.model.LSTMPCellWithClip(input_size=3, hidden_size=5, projection_size=3, projection_clip=3, cell_clip=3)
# model.initialize(init=mx.initializer.One())
# model.hybridize()
# inputs = mx.nd.ones(shape=(1,3))
# begin_states = model.begin_state(batch_size=1)
# outputs = model(inputs, begin_states)
# print(outputs[0])
# print(outputs[1][0])
# print(outputs[1][1])