import json
import logging
from typing import Union, List, Dict, Any
import warnings

import mxnet as mx
import gluonnlp as nlp

from mxnet import gluon, autograd, nd

import numpy

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import h5py
except ImportError:
    import h5py


# class Elmo(torch.nn.Module):
#     """
#     Compute ELMo representations using a pre-trained bidirectional language model.
#
#     See "Deep contextualized word representations", Peters et al. for details.
#
#     This module takes character id input and computes ``num_output_representations`` different layers
#     of ELMo representations.  Typically ``num_output_representations`` is 1 or 2.  For example, in
#     the case of the SRL model in the above paper, ``num_output_representations=1`` where ELMo was included at
#     the input token representation layer.  In the case of the SQuAD model, ``num_output_representations=2``
#     as ELMo was also included at the GRU output layer.
#
#     In the implementation below, we learn separate scalar weights for each output layer,
#     but only run the biLM once on each input sequence for efficiency.
#
#     Parameters
#     ----------
#     options_file : ``str``, required.
#         ELMo JSON options file
#     weight_file : ``str``, required.
#         ELMo hdf5 weight file
#     num_output_representations: ``int``, required.
#         The number of ELMo representation layers to output.
#     requires_grad: ``bool``, optional
#         If True, compute gradient of ELMo parameters for fine tuning.
#     do_layer_norm : ``bool``, optional, (default=False).
#         Should we apply layer normalization (passed to ``ScalarMix``)?
#     dropout : ``float``, optional, (default = 0.5).
#         The dropout to be applied to the ELMo representations.
#     vocab_to_cache : ``List[str]``, optional, (default = 0.5).
#         A list of words to pre-compute and cache character convolutions
#         for. If you use this option, Elmo expects that you pass word
#         indices of shape (batch_size, timesteps) to forward, instead
#         of character indices. If you use this option and pass a word which
#         wasn't pre-cached, this will break.
#     keep_sentence_boundaries : ``bool``, optional, (default=False)
#         If True, the representation of the sentence boundary tokens are
#         not removed.
#     scalar_mix_parameters : ``List[int]``, optional, (default=None)
#         If not ``None``, use these scalar mix parameters to weight the representations
#         produced by different layers. These mixing weights are not updated during
#         training.
#     module : ``torch.nn.Module``, optional, (default = None).
#         If provided, then use this module instead of the pre-trained ELMo biLM.
#         If using this option, then pass ``None`` for both ``options_file``
#         and ``weight_file``.  The module must provide a public attribute
#         ``num_layers`` with the number of internal layers and its ``forward``
#         method must return a ``dict`` with ``activations`` and ``mask`` keys
#         (see `_ElmoBilm`` for an example).  Note that ``requires_grad`` is also
#         ignored with this option.
#     """
#     def __init__(self,
#                  options_file: str,
#                  weight_file: str,
#                  num_output_representations: int,
#                  requires_grad: bool = False,
#                  do_layer_norm: bool = False,
#                  dropout: float = 0.5,
#                  vocab_to_cache: List[str] = None,
#                  keep_sentence_boundaries: bool = False,
#                  scalar_mix_parameters: List[float] = None,
#                  module: torch.nn.Module = None) -> None:
#         super(Elmo, self).__init__()
#
#         logging.info("Initializing ELMo")
#         if module is not None:
#             if options_file is not None or weight_file is not None:
#                 raise ConfigurationError(
#                         "Don't provide options_file or weight_file with module")
#             self._elmo_lstm = module
#         else:
#             self._elmo_lstm = _ElmoBiLm(options_file,
#                                         weight_file,
#                                         requires_grad=requires_grad,
#                                         vocab_to_cache=vocab_to_cache)
#         self._has_cached_vocab = vocab_to_cache is not None
#         self._keep_sentence_boundaries = keep_sentence_boundaries
#         self._dropout = Dropout(p=dropout)
#         self._scalar_mixes: Any = []
#         for k in range(num_output_representations):
#             scalar_mix = ScalarMix(
#                     self._elmo_lstm.num_layers,
#                     do_layer_norm=do_layer_norm,
#                     initial_scalar_parameters=scalar_mix_parameters,
#                     trainable=scalar_mix_parameters is None)
#             self.add_module('scalar_mix_{}'.format(k), scalar_mix)
#             self._scalar_mixes.append(scalar_mix)
#
#     def get_output_dim(self):
#         return self._elmo_lstm.get_output_dim()
#
#     def forward(self,    # pylint: disable=arguments-differ
#                 inputs: torch.Tensor,
#                 word_inputs: torch.Tensor = None) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
#         """
#         Parameters
#         ----------
#         inputs: ``torch.Tensor``, required.
#         Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
#         word_inputs : ``torch.Tensor``, required.
#             If you passed a cached vocab, you can in addition pass a tensor of shape
#             ``(batch_size, timesteps)``, which represent word ids which have been pre-cached.
#
#         Returns
#         -------
#         Dict with keys:
#         ``'elmo_representations'``: ``List[torch.Tensor]``
#             A ``num_output_representations`` list of ELMo representations for the input sequence.
#             Each representation is shape ``(batch_size, timesteps, embedding_dim)``
#         ``'mask'``:  ``torch.Tensor``
#             Shape ``(batch_size, timesteps)`` long tensor with sequence mask.
#         """
#         # reshape the input if needed
#         original_shape = inputs.size()
#         if len(original_shape) > 3:
#             timesteps, num_characters = original_shape[-2:]
#             reshaped_inputs = inputs.view(-1, timesteps, num_characters)
#         else:
#             reshaped_inputs = inputs
#
#         if word_inputs is not None:
#             original_word_size = word_inputs.size()
#             if self._has_cached_vocab and len(original_word_size) > 2:
#                 reshaped_word_inputs = word_inputs.view(-1, original_word_size[-1])
#             elif not self._has_cached_vocab:
#                 logger.warning("Word inputs were passed to ELMo but it does not have a cached vocab.")
#                 reshaped_word_inputs = None
#             else:
#                 reshaped_word_inputs = word_inputs
#         else:
#             reshaped_word_inputs = word_inputs
#
#         # run the biLM
#         bilm_output = self._elmo_lstm(reshaped_inputs, reshaped_word_inputs)
#         layer_activations = bilm_output['activations']
#         mask_with_bos_eos = bilm_output['mask']
#
#         # compute the elmo representations
#         representations = []
#         for i in range(len(self._scalar_mixes)):
#             scalar_mix = getattr(self, 'scalar_mix_{}'.format(i))
#             representation_with_bos_eos = scalar_mix(layer_activations, mask_with_bos_eos)
#             if self._keep_sentence_boundaries:
#                 processed_representation = representation_with_bos_eos
#                 processed_mask = mask_with_bos_eos
#             else:
#                 representation_without_bos_eos, mask_without_bos_eos = remove_sentence_boundaries(
#                         representation_with_bos_eos, mask_with_bos_eos)
#                 processed_representation = representation_without_bos_eos
#                 processed_mask = mask_without_bos_eos
#             representations.append(self._dropout(processed_representation))
#
#         # reshape if necessary
#         if word_inputs is not None and len(original_word_size) > 2:
#             mask = processed_mask.view(original_word_size)
#             elmo_representations = [representation.view(original_word_size + (-1, ))
#                                     for representation in representations]
#         elif len(original_shape) > 3:
#             mask = processed_mask.view(original_shape[:-1])
#             elmo_representations = [representation.view(original_shape[:-1] + (-1, ))
#                                     for representation in representations]
#         else:
#             mask = processed_mask
#             elmo_representations = representations
#
#         return {'elmo_representations': elmo_representations, 'mask': mask}
#
#     # The add_to_archive logic here requires a custom from_params.
#     @classmethod
#     def from_params(cls, params: Params) -> 'Elmo':
#         # Add files to archive
#         params.add_file_to_archive('options_file')
#         params.add_file_to_archive('weight_file')
#
#         options_file = params.pop('options_file')
#         weight_file = params.pop('weight_file')
#         requires_grad = params.pop('requires_grad', False)
#         num_output_representations = params.pop('num_output_representations')
#         do_layer_norm = params.pop_bool('do_layer_norm', False)
#         keep_sentence_boundaries = params.pop_bool('keep_sentence_boundaries', False)
#         dropout = params.pop_float('dropout', 0.5)
#         params.assert_empty(cls.__name__)
#
#         return cls(options_file=options_file,
#                    weight_file=weight_file,
#                    num_output_representations=num_output_representations,
#                    requires_grad=requires_grad,
#                    do_layer_norm=do_layer_norm,
#                    keep_sentence_boundaries=keep_sentence_boundaries,
#                    dropout=dropout)


# def batch_to_ids(batch: List[List[str]]) -> torch.Tensor:
#     """
#     Converts a batch of tokenized sentences to a tensor representing the sentences with encoded characters
#     (len(batch), max sentence length, max word length).
#
#     Parameters
#     ----------
#     batch : ``List[List[str]]``, required
#         A list of tokenized sentences.
#
#     Returns
#     -------
#         A tensor of padded character ids.
#     """
#     instances = []
#     indexer = ELMoTokenCharactersIndexer()
#     for sentence in batch:
#         tokens = [Token(token) for token in sentence]
#         field = TextField(tokens,
#                           {'character_ids': indexer})
#         instance = Instance({"elmo": field})
#         instances.append(instance)
#
#     dataset = Batch(instances)
#     vocab = Vocabulary()
#     dataset.index_instances(vocab)
#     return dataset.as_tensor_dict()['elmo']['character_ids']

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

    bos_token = '<S>'
    eos_token = '</S>'


def add_sentence_boundary_token_ids(tensor,
                                    mask,
                                    sentence_begin_token,
                                    sentence_end_token):
    """
    Add begin/end of sentence tokens to the batch of sentences.
    Given a batch of sentences with size ``(batch_size, timesteps)`` or
    ``(batch_size, timesteps, dim)`` this returns a tensor of shape
    ``(batch_size, timesteps + 2)`` or ``(batch_size, timesteps + 2, dim)`` respectively.

    Returns both the new tensor and updated mask.

    Parameters
    ----------
    tensor : ``torch.Tensor``
        A tensor of shape ``(batch_size, timesteps)`` or ``(batch_size, timesteps, dim)``
    mask : ``torch.Tensor``
         A tensor of shape ``(batch_size, timesteps)``
    sentence_begin_token: Any (anything that can be broadcast in torch for assignment)
        For 2D input, a scalar with the <S> id. For 3D input, a tensor with length dim.
    sentence_end_token: Any (anything that can be broadcast in torch for assignment)
        For 2D input, a scalar with the </S> id. For 3D input, a tensor with length dim.

    Returns
    -------
    tensor_with_boundary_tokens : ``torch.Tensor``
        The tensor with the appended and prepended boundary tokens. If the input was 2D,
        it has shape (batch_size, timesteps + 2) and if the input was 3D, it has shape
        (batch_size, timesteps + 2, dim).
    new_mask : ``torch.Tensor``
        The new mask for the tensor, taking into account the appended tokens
        marking the beginning and end of the sentence.
    """
    # TODO: matthewp, profile this transfer
    sequence_lengths = mask.sum(axis=1).asnumpy()
    tensor_shape = list(tensor.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] + 2
    tensor_with_boundary_tokens = nd.zeros(new_shape)
    if len(tensor_shape) == 2:
        tensor_with_boundary_tokens[:, 1:-1] = tensor
        tensor_with_boundary_tokens[:, 0] = sentence_begin_token
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[i, j + 1] = sentence_end_token
        new_mask = tensor_with_boundary_tokens != 0
    elif len(tensor_shape) == 3:
        tensor_with_boundary_tokens[:, 1:-1, :] = tensor
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[i, 0, :] = sentence_begin_token
            tensor_with_boundary_tokens[i, int(j + 1), :] = sentence_end_token
        new_mask = (tensor_with_boundary_tokens > 0).sum(axis=-1) > 0
    else:
        raise ValueError("add_sentence_boundary_token_ids only accepts 2D and 3D input")

    return tensor_with_boundary_tokens, new_mask

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
                 requires_grad=False):
        super(_ElmoCharacterEncoder, self).__init__()

        with open(options_file, 'r') as fin:
            self._options = json.load(fin)
        self._weight_file = weight_file
        self._requires_grad = requires_grad

        self.output_dim = self._options['lstm']['projection_dim']

        #TODO
        # self._load_weights()

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

        self._char_embedding_weights = nd.ones(shape=(ELMoCharacterMapper.padding_character+2, self._char_embed_dim))

        with self.name_scope():
            self._convolutions = gluon.nn.HybridSequential()
            for i, (width, num) in enumerate(self._filters):
                conv = gluon.nn.Conv1D(
                    in_channels=self._char_embed_dim,
                    channels=num,
                    kernel_size=width,
                    use_bias=True
                )
                self._convolutions.add(conv)
            self._highways = nlp.model.Highway(input_size=self._n_filters, num_layers=self._n_highway,
                                               activation='relu')
            self._projection = gluon.nn.Dense(in_units=self._n_filters, units=self.output_dim,
                                              use_bias=True)

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
        character_embedding = nd.Embedding(
                character_ids_with_bos_eos.reshape(-1, max_chars_per_token),
                self._char_embedding_weights,
                input_dim=ELMoCharacterMapper.padding_character+2,
                output_dim=self._char_embedding_weights.shape[1]
        )

        # run convolutions
        cnn_options = self._options['char_cnn']
        if cnn_options['activation'] == 'tanh':
            activation = gluon.nn.Activation('tanh')
        elif cnn_options['activation'] == 'relu':
            activation = gluon.nn.Activation('relu')
        else:
            raise NotImplementedError

        # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        character_embedding = nd.transpose(character_embedding, axes=(0, 2, 1))
        convs = []
        for _, conv in enumerate(self._convolutions):
            convolved = conv(character_embedding)
            # (batch_size * sequence_length, n_filters for this width)
            convolved = nd.max(convolved, axis=-1)
            convolved = activation(convolved)
            convs.append(convolved)

        # (batch_size * sequence_length, n_filters)
        token_embedding = nd.concat(*convs, dim=-1)

        # apply the highway layers (batch_size * sequence_length, n_filters)
        token_embedding = self._highways(token_embedding)

        # final projection  (batch_size * sequence_length, embedding_dim)
        token_embedding = self._projection(token_embedding)

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

        weights = numpy.zeros(
                (char_embed_weights.shape[0] + 1, char_embed_weights.shape[1]),
                dtype='float32'
        )
        weights[1:, :] = char_embed_weights
        self._char_embedding_weights = nd.array(weights)
        #TODO:
        # self._char_embedding_weights = gluon.Parameter(name='char_embed', shape=weights.shape)
        # self._char_embedding_weights.grad_req = self._requires_grad
        # self._char_embedding_weights.set_data(nd.array(weights))

    def _load_cnn_weights(self):
        #TODO:
        # cnn_options = self._options['char_cnn']
        # filters = cnn_options['filters']
        # char_embed_dim = cnn_options['embedding']['dim']

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

        #TODO:
        # convolutions = gluon.nn.HybridSequential()
        # for i, (width, num) in enumerate(filters):
        #     conv = gluon.nn.Conv1d(
        #         in_channels=char_embed_dim,
        #         channels=num,
        #         kernel_size=width,
        #         use_bias=True
        #     )
        #     # load the weights
        #     with h5py.File(self._weight_file, 'r') as fin:
        #         weight = fin['CNN']['W_cnn_{}'.format(i)][...]
        #         bias = fin['CNN']['b_cnn_{}'.format(i)][...]
        #
        #     w_reshaped = numpy.transpose(weight.squeeze(axis=0), axes=(2, 1, 0))
        #     if w_reshaped.shape != tuple(conv.weight.data().shape):
        #         raise ValueError("Invalid weight file")
        #     conv.weight.data()[:] = nd.array(w_reshaped)
        #     conv.bias.data()[:] = nd.array(bias)
        #
        #     conv.weight.grad_req = self.requires_grad
        #     conv.bias.grad_req = self.requires_grad
        #
        #     convolutions.add(conv)
        #
        # self._convolutions = convolutions

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

    def _load_projection(self):
        #TODO
        # cnn_options = self._options['char_cnn']
        # filters = cnn_options['filters']
        # n_filters = sum(f[1] for f in filters)
        #
        # self._projection = gluon.nn.Dense(in_units=n_filters, units=self.output_dim, use_bias=True)
        with h5py.File(self._weight_file, 'r') as fin:
            weight = fin['CNN_proj']['W_proj'][...]
            bias = fin['CNN_proj']['b_proj'][...]
            self._projection.weight.data()[:] = nd.array(numpy.transpose(weight))
            self._projection.bias.data()[:] = nd.array(bias)

            self._projection.weight.grad_req = self._requires_grad
            self._projection.bias.grad_req = self._requires_grad


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
                 weight_file,
                 requires_grad='null',
                 vocab_to_cache=None):
        super(_ElmoBiLm, self).__init__()

        self._options_file = options_file

        self._weight_file = weight_file

        self._token_embedder = _ElmoCharacterEncoder(options_file, weight_file, requires_grad=requires_grad)

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
        #TODO: cache
        # if vocab_to_cache:
        #     logging.info("Caching character cnn layers for words in vocabulary.")
        #     # This sets 3 attributes, _word_embedding, _bos_embedding and _eos_embedding.
        #     # They are set in the method so they can be accessed from outside the
        #     # constructor.
        #     self.create_cached_cnn_embeddings(vocab_to_cache)

        with open(options_file, 'r') as fin:
            options = json.load(fin)
        #TODO: add residual connection
        if not options['lstm'].get('use_skip_connections'):
            # raise ConfigurationError('We only support pretrained biLMs with residual connections')
            raise NotImplementedError
        #TODO: import right model
        # self._elmo_lstm = ElmoLstm(input_size=options['lstm']['projection_dim'],
        #                            hidden_size=options['lstm']['projection_dim'],
        #                            cell_size=options['lstm']['dim'],
        #                            num_layers=options['lstm']['n_layers'],
        #                            memory_cell_clip_value=options['lstm']['cell_clip'],
        #                            state_projection_clip_value=options['lstm']['proj_clip'],
        #                            requires_grad=requires_grad)
        #TODO: check proj_size
        with self.name_scope():
            self._elmo_lstm = nlp.model.BiLMEncoder(mode='lstmpc',
                                                    input_size=options['lstm']['projection_dim'],
                                                    hidden_size=options['lstm']['dim'],
                                                    proj_size=options['lstm']['projection_dim'],
                                                    num_layers=options['lstm']['n_layers'],
                                                    cell_clip=options['lstm']['cell_clip'],
                                                    proj_clip=options['lstm']['proj_clip'])
        #TODO:
        # self._elmo_lstm.collect_params().setattr('grad_req', 'null')
        # #TODO: load weights implementation
        # self._elmo_lstm.load_weights(weight_file)
        # # Number of representation layers including context independent layer
        # self.num_layers = options['lstm']['n_layers'] + 1

    def get_output_dim(self):
        return 2 * self._token_embedder.get_output_dim()

    def forward(self,  # pylint: disable=arguments-differ
                inputs,
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
            # mask = token_embedding['mask']
            # type_representation = token_embedding['token_embedding']
        lstm_outputs = self._elmo_lstm(type_representation, None, mask)

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
        return lstm_outputs, mask

    # def create_cached_cnn_embeddings(self, tokens: List[str]) -> None:
    #     """
    #     Given a list of tokens, this method precomputes word representations
    #     by running just the character convolutions and highway layers of elmo,
    #     essentially creating uncontextual word vectors. On subsequent forward passes,
    #     the word ids are looked up from an embedding, rather than being computed on
    #     the fly via the CNN encoder.
    #
    #     This function sets 3 attributes:
    #
    #     _word_embedding : ``torch.Tensor``
    #         The word embedding for each word in the tokens passed to this method.
    #     _bos_embedding : ``torch.Tensor``
    #         The embedding for the BOS token.
    #     _eos_embedding : ``torch.Tensor``
    #         The embedding for the EOS token.
    #
    #     Parameters
    #     ----------
    #     tokens : ``List[str]``, required.
    #         A list of tokens to precompute character convolutions for.
    #     """
    #     tokens = [ELMoCharacterMapper.bos_token, ELMoCharacterMapper.eos_token] + tokens
    #     timesteps = 32
    #     batch_size = 32
    #     chunked_tokens = lazy_groups_of(iter(tokens), timesteps)
    #
    #     all_embeddings = []
    #     device = get_device_of(next(self.parameters()))
    #     for batch in lazy_groups_of(chunked_tokens, batch_size):
    #         # Shape (batch_size, timesteps, 50)
    #         batched_tensor = batch_to_ids(batch)
    #         # NOTE: This device check is for when a user calls this method having
    #         # already placed the model on a device. If this is called in the
    #         # constructor, it will probably happen on the CPU. This isn't too bad,
    #         # because it's only a few convolutions and will likely be very fast.
    #         if device >= 0:
    #             batched_tensor = batched_tensor.cuda(device)
    #         output = self._token_embedder(batched_tensor)
    #         token_embedding = output["token_embedding"]
    #         mask = output["mask"]
    #         token_embedding, _ = remove_sentence_boundaries(token_embedding, mask)
    #         all_embeddings.append(token_embedding.view(-1, token_embedding.size(-1)))
    #     full_embedding = torch.cat(all_embeddings, 0)
    #
    #     # We might have some trailing embeddings from padding in the batch, so
    #     # we clip the embedding and lookup to the right size.
    #     full_embedding = full_embedding[:len(tokens), :]
    #     embedding = full_embedding[2:len(tokens), :]
    #     vocab_size, embedding_dim = list(embedding.size())
    #
    #     from allennlp.modules.token_embedders import Embedding # type: ignore
    #     self._bos_embedding = full_embedding[0, :]
    #     self._eos_embedding = full_embedding[1, :]
    #     self._word_embedding = Embedding(vocab_size, # type: ignore
    #                                      embedding_dim,
    #                                      weight=embedding.data,
    #                                      trainable=self._requires_grad,
    #                                      padding_index=0)

model = _ElmoBiLm(options_file='elmo_2x1024_128_2048cnn_1xhighway_options.json',
                  weight_file='elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
model.initialize(mx.init.Xavier(), ctx=mx.cpu(0))
model._token_embedder._load_weights()
model._elmo_lstm.load_weights(model._weight_file, model._requires_grad)
inputs = nd.ones(shape=(20,35,50))
outputs, mask = model(inputs)
print('outputs:')
print(outputs)
print('mask:')
print(mask)

model.save_parameters('elmo_2x1024_128_2048cnn_1xhighway_weights.params')

