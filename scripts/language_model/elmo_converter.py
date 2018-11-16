import json
import logging
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


# class Add_Sentence_Boundary_Token_IDs(object):
#     """
#     Add begin/end of sentence tokens to the batch of sentences.
#     Given a batch of sentences with size ``(batch_size, seq_len)`` or
#     ``(batch_size, seq_len, dim)`` this returns a tensor of shape
#     ``(batch_size, seq_len + 2)`` or ``(batch_size, seq_len + 2, dim)`` respectively.
#
#     Returns both the new tensor and updated mask.
#
#     Parameters
#     ----------
#     tensor : ``torch.Tensor``
#         A tensor of shape ``(batch_size, timesteps)`` or ``(batch_size, timesteps, dim)``
#     mask : ``torch.Tensor``
#          A tensor of shape ``(batch_size, timesteps)``
#     sentence_begin_token: Any (anything that can be broadcast in torch for assignment)
#         For 2D input, a scalar with the <S> id. For 3D input, a tensor with length dim.
#     sentence_end_token: Any (anything that can be broadcast in torch for assignment)
#         For 2D input, a scalar with the </S> id. For 3D input, a tensor with length dim.
#
#     Returns
#     -------
#     tensor_with_boundary_tokens : ``torch.Tensor``
#         The tensor with the appended and prepended boundary tokens. If the input was 2D,
#         it has shape (batch_size, timesteps + 2) and if the input was 3D, it has shape
#         (batch_size, timesteps + 2, dim).
#     new_mask : ``torch.Tensor``
#         The new mask for the tensor, taking into account the appended tokens
#         marking the beginning and end of the sentence.
#     """
#     def __init__(self):
#         super(Add_Sentence_Boundary_Token_IDs, self).__init__()
#
#
#     def __call__(self, inputs, mask, sentence_begin_token, sentence_end_token):
#         #TODO
#         sequence_lengths = mask.sum(axis=1)
#         # sequence_lengths = mask.sum(axis=1)
#         inputs_shape = list(inputs.shape)
#         new_shape = list(inputs_shape)
#         new_shape[1] = inputs_shape[1] + 2
#         inputs_with_boundary_tokens = nd.zeros(new_shape)
#         if len(inputs_shape) == 2:
#             inputs_with_boundary_tokens[:, 1:-1] = inputs
#             inputs_with_boundary_tokens[:, 0] = sentence_begin_token
#             for i, j in enumerate(sequence_lengths):
#                 inputs_with_boundary_tokens[i, j + 1] = sentence_end_token
#             new_mask = inputs_with_boundary_tokens != 0
#         elif len(inputs_shape) == 3:
#             inputs_with_boundary_tokens[:, 1:-1, :] = inputs
#             for i, j in enumerate(sequence_lengths):
#                 inputs_with_boundary_tokens[i, 0, :] = sentence_begin_token
#                 inputs_with_boundary_tokens[i, int(j + 1), :] = sentence_end_token
#             new_mask = (inputs_with_boundary_tokens > 0).sum(axis=-1) > 0
#         else:
#             raise NotImplementedError
#         return inputs_with_boundary_tokens, new_mask

def add_sentence_boundary_token_ids(inputs, mask, sentence_begin_token, sentence_end_token):

    #TODO
    sequence_lengths = mask.sum(axis=1).asnumpy()
    # sequence_lengths = mask.sum(axis=1)
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
        cnn_options = self._options['char_cnn']
        if cnn_options['activation'] == 'tanh':
            conv_layer_activation = 'tanh'
        elif cnn_options['activation'] == 'relu':
            conv_layer_activation = 'relu'
        else:
            raise NotImplementedError

        self._char_embedding_weights = nd.ones(shape=(ELMoCharacterMapper.padding_character+2, self._char_embed_dim))
        # self._load_char_embedding()

        # self._add_sentence_boundary_token_ids = Add_Sentence_Boundary_Token_IDs()

        with self.name_scope():
            # character_embedding = nd.Embedding(
            #     character_ids_with_bos_eos.reshape(-1, max_chars_per_token),
            #     self._char_embedding_weights,
            #     input_dim=ELMoCharacterMapper.padding_character + 2,
            #     output_dim=self._char_embedding_weights.shape[1]
            # )
            self._char_embedding = gluon.nn.Embedding(ELMoCharacterMapper.padding_character+2,
                                                      self._char_embed_dim)
            if self._cnn_encoder == 'manual':
                self._convolutions = gluon.nn.HybridSequential()
                for i, (width, num) in enumerate(self._filters):
                    conv = gluon.nn.Conv1D(
                        in_channels=self._char_embed_dim,
                        channels=num,
                        kernel_size=width,
                        use_bias=True
                    )
                    self._convolutions.add(conv)
                self._highways = nlp.model.Highway(input_size=self._n_filters,
                                                   num_layers=self._n_highway,
                                                   activation='relu',
                                                   highway_bias=nlp.initializer.HighwayBias(
                                                       nonlinear_transform_bias=0.0,
                                                       transform_gate_bias=1.0))
                self._projection = gluon.nn.Dense(in_units=self._n_filters, units=self.output_dim,
                                                  use_bias=True)
            elif self._cnn_encoder == 'encoder':
                ngram_filter_sizes = []
                num_filters = []
                for _, (width, num) in enumerate(self._filters):
                    ngram_filter_sizes.append(width)
                    num_filters.append(num)
                self._convolutions = nlp.model.ConvolutionalEncoder(embed_size=self._char_embed_dim,
                                                                    num_filters=tuple(num_filters),
                                                                    ngram_filter_sizes=tuple(ngram_filter_sizes),
                                                                    conv_layer_activation=conv_layer_activation,
                                                                    num_highway=self._n_highway,
                                                                    highway_bias=nlp.initializer.HighwayBias(
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
        # character_embedding = nd.Embedding(
        #         character_ids_with_bos_eos.reshape(-1, max_chars_per_token),
        #         self._char_embedding_weights,
        #         input_dim=ELMoCharacterMapper.padding_character+2,
        #         output_dim=self._char_embedding_weights.shape[1]
        # )
        character_embedding = self._char_embedding(character_ids_with_bos_eos.reshape(-1, max_chars_per_token))

        if self._cnn_encoder == 'manual':
            # run convolutions
            cnn_options = self._options['char_cnn']
            if cnn_options['activation'] == 'tanh':
                activation = gluon.nn.Activation('tanh')
            elif cnn_options['activation'] == 'relu':
                activation = gluon.nn.Activation('relu')
            else:
                raise NotImplementedError

            # print(character_embedding[:1, :1, :5])
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
        elif self._cnn_encoder == 'encoder':
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
                 lstm_mode='manual',
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
            if lstm_mode == 'manual':
                self._elmo_lstm = nlp.model.BiLMEncoder(mode='lstmpc',
                                                        input_size=options['lstm']['projection_dim'],
                                                        hidden_size=options['lstm']['dim'],
                                                        proj_size=options['lstm']['projection_dim'],
                                                        num_layers=options['lstm']['n_layers'],
                                                        cell_clip=options['lstm']['cell_clip'],
                                                        proj_clip=options['lstm']['proj_clip'])
            elif lstm_mode == 'unroll':
                self._elmo_lstm = nlp.model.BiLMEncoderUnroll(mode='lstmpc',
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
            # mask = token_embedding['mask']
            # type_representation = token_embedding['token_embedding']
        lstm_outputs = self._elmo_lstm(type_representation, states, mask)

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


# print('0!!!')
# model0 = _ElmoBiLm(options_file='elmo_2x1024_128_2048cnn_1xhighway_options.json',
#                    weight_file='elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5',
#                    lstm_mode='manual')
# print(model0)
# model0.initialize(ctx=mx.cpu(0))
# model0._token_embedder._load_weights()
# model0._token_embedder._char_embedding.weight.set_data(model0._token_embedder._char_embedding_weights)
# model0._elmo_lstm.load_weights(model0._weight_file, model0._requires_grad)
# model0.save_parameters('elmo_2x1024_128_2048cnn_1xhighway_weights.params')
# inputs = nd.ones(shape=(20,35,50))
# outputs0, mask0 = model0(inputs)
# print_outputs(outputs0)

# print('1!!!')
# model1 = _ElmoBiLm(options_file='elmo_2x1024_128_2048cnn_1xhighway_options.json', lstm_mode='manual')
# print(model1)
# model1.load_parameters('elmo_2x1024_128_2048cnn_1xhighway_weights.params')
# inputs = nd.ones(shape=(20,35,50))
# outputs1, mask1 = model1(inputs)
# print_outputs(outputs1)
#
# print('2!!!')
# model2 = _ElmoBiLm(options_file='elmo_2x1024_128_2048cnn_1xhighway_options.json',
#                    weight_file='elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5',
#                    lstm_mode='unroll')
# print(model2)
# model2.initialize(ctx=mx.cpu(0))
# model2._token_embedder._load_weights()
# model2._token_embedder._char_embedding.weight.set_data(model2._token_embedder._char_embedding_weights)
# model2._elmo_lstm.load_weights(model2._weight_file, model2._requires_grad)
# model2.save_parameters('elmo_2x1024_128_2048cnn_1xhighway_weights.unroll.params')
# inputs = nd.ones(shape=(20,35,50))
# outputs2, mask2 = model2(inputs)
# print_outputs(outputs2)
#
#
# print('3!!!')
# model3 = _ElmoBiLm(options_file='elmo_2x1024_128_2048cnn_1xhighway_options.json', lstm_mode='unroll')
# print(model3)
# model3.load_parameters('elmo_2x1024_128_2048cnn_1xhighway_weights.unroll.params')
# inputs = nd.ones(shape=(20,35,50))
# outputs3, mask3 = model3(inputs)
# print_outputs(outputs3)

data_dir = '/Users/chgwang/Documents/code/elmo-data/'
options_file = 'elmo_2x4096_512_2048cnn_2xhighway_options.json'
weight_file = 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

print('4!!!')
model4 = _ElmoBiLm(options_file=data_dir + options_file,
                   weight_file=data_dir + weight_file,
                   lstm_mode='manual',
                   cnn_encoder='encoder')
print(model4)
model4.initialize()
model4._token_embedder._load_weights()
model4._token_embedder._char_embedding.weight.set_data(model4._token_embedder._char_embedding_weights)
model4._elmo_lstm.load_weights(model4._weight_file, model4._requires_grad)
model4.save_parameters(data_dir + weight_file + '.params')
model4.hybridize()
inputs = nd.ones(shape=(20,35,50))
begin_state = model4._elmo_lstm.begin_state(mx.nd.zeros, batch_size=20)
outputs4, mask4 = model4(inputs, begin_state)
print_outputs_nd(outputs4[0])

print('5!!!')
model5 = _ElmoBiLm(options_file=data_dir + options_file,
                   lstm_mode='manual',
                   cnn_encoder='encoder')
print(model5)
model5.load_parameters(data_dir + weight_file + '.params')
model5.hybridize()
inputs = nd.ones(shape=(20,35,50))
begin_state = model5._elmo_lstm.begin_state(mx.nd.zeros, batch_size=20)
outputs5, mask5 = model5(inputs, begin_state)
print_outputs_nd(outputs5[0])

from numpy.testing import assert_almost_equal
import torch
t_output = torch.load(data_dir + weight_file + '.tout')
assert_almost_equal(outputs4[0].asnumpy(), t_output.numpy(), decimal=5)
assert_almost_equal(outputs5[0].asnumpy(), t_output.numpy(), decimal=5)
print('equal with torch!')

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