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

# pylint: disable=consider-iterating-dictionary

"""Vocabulary."""
from __future__ import absolute_import
from __future__ import print_function

__all__ = ['Vocab', 'UnicodeCharsVocabulary']

import json
import warnings

from mxnet import nd
import numpy as np

from .data.utils import DefaultLookupDict
from . import _constants as C
from . import embedding as emb


class Vocab(object):
    """Indexing and embedding attachment for text tokens.

    Parameters
    ----------
    counter : Counter or None, default None
        Counts text token frequencies in the text data. Its keys will be indexed according to
        frequency thresholds such as `max_size` and `min_freq`. Keys of `counter`,
        `unknown_token`, and values of `reserved_tokens` must be of the same hashable type.
        Examples: str, int, and tuple.
    max_size : None or int, default None
        The maximum possible number of the most frequent tokens in the keys of `counter` that can be
        indexed. Note that this argument does not count any token from `reserved_tokens`. Suppose
        that there are different keys of `counter` whose frequency are the same, if indexing all of
        them will exceed this argument value, such keys will be indexed one by one according to
        their __cmp__() order until the frequency threshold is met. If this argument is None or
        larger than its largest possible value restricted by `counter` and `reserved_tokens`, this
        argument has no effect.
    min_freq : int, default 1
        The minimum frequency required for a token in the keys of `counter` to be indexed.
    unknown_token : hashable object or None, default '<unk>'
        The representation for any unknown token. In other words, any unknown token will be indexed
        as the same representation. If None, looking up an unknown token will result in KeyError.
    padding_token : hashable object or None, default '<pad>'
        The representation for the special token of padding token.
    bos_token : hashable object or None, default '<bos>'
        The representation for the special token of beginning-of-sequence token.
    eos_token : hashable object or None, default '<eos>'
        The representation for the special token of end-of-sequence token.
    reserved_tokens : list of hashable objects or None, default None
        A list of reserved tokens (excluding `unknown_token`) that will always be indexed, such as
        special symbols representing padding, beginning of sentence, and end of sentence. It cannot
        contain `unknown_token` or duplicate reserved tokens. Keys of `counter`, `unknown_token`,
        and values of `reserved_tokens` must be of the same hashable type. Examples: str, int, and
        tuple.

    Properties
    ----------
    embedding : instance of :class:`gluonnlp.embedding.TokenEmbedding`
        The embedding of the indexed tokens.
    idx_to_token : list of strs
        A list of indexed tokens where the list indices and the token indices are aligned.
    reserved_tokens : list of strs or None
        A list of reserved tokens that will always be indexed.
    token_to_idx : dict mapping str to int
        A dict mapping each token to its index integer.
    unknown_token : hashable object or None
        The representation for any unknown token. In other words, any unknown token will be indexed
        as the same representation.


    Examples
    --------

    >>> text_data = " hello world \\\\n hello nice world \\\\n hi world \\\\n"
    >>> counter = gluonnlp.data.count_tokens(text_data)
    >>> my_vocab = gluonnlp.Vocab(counter)
    >>> fasttext = gluonnlp.embedding.create('fasttext', source='wiki.simple.vec')
    >>> my_vocab.set_embedding(fasttext)
    >>> my_vocab.embedding[['hello', 'world']]
    [[  3.95669997e-01   2.14540005e-01  -3.53889987e-02  -2.42990002e-01
        ...
       -7.54180014e-01  -3.14429998e-01   2.40180008e-02  -7.61009976e-02]
     [  1.04440004e-01  -1.08580001e-01   2.72119999e-01   1.32990003e-01
        ...
       -3.73499990e-01   5.67310005e-02   5.60180008e-01   2.90190000e-02]]
    <NDArray 2x300 @cpu(0)>

    >>> my_vocab[['hello', 'world']]
    [5, 4]

    >>> input_dim, output_dim = my_vocab.embedding.idx_to_vec.shape
    >>> layer = gluon.nn.Embedding(input_dim, output_dim)
    >>> layer.initialize()
    >>> layer.weight.set_data(my_vocab.embedding.idx_to_vec)
    >>> layer(nd.array([5, 4]))
    [[  3.95669997e-01   2.14540005e-01  -3.53889987e-02  -2.42990002e-01
        ...
       -7.54180014e-01  -3.14429998e-01   2.40180008e-02  -7.61009976e-02]
     [  1.04440004e-01  -1.08580001e-01   2.72119999e-01   1.32990003e-01
        ...
       -3.73499990e-01   5.67310005e-02   5.60180008e-01   2.90190000e-02]]
    <NDArray 2x300 @cpu(0)>

    >>> glove = gluonnlp.embedding.create('glove', source='glove.6B.50d.txt')
    >>> my_vocab.set_embedding(glove)
    >>> my_vocab.embedding[['hello', 'world']]
    [[  -0.38497001  0.80092001
        ...
        0.048833    0.67203999]
     [  -0.41486001  0.71847999
        ...
       -0.37639001 -0.67541999]]
    <NDArray 2x50 @cpu(0)>

    """

    def __init__(self, counter=None, max_size=None, min_freq=1, unknown_token=C.UNK_TOKEN,
                 padding_token=C.PAD_TOKEN, bos_token=C.BOS_TOKEN, eos_token=C.EOS_TOKEN,
                 reserved_tokens=None):

        # Sanity checks.
        assert min_freq > 0, '`min_freq` must be set to a positive value.'

        self._unknown_token = unknown_token
        special_tokens = []
        self._padding_token = padding_token
        if padding_token:
            special_tokens.append(padding_token)
        self._bos_token = bos_token
        if bos_token:
            special_tokens.append(bos_token)
        self._eos_token = eos_token
        if eos_token:
            special_tokens.append(eos_token)
        if reserved_tokens:
            special_tokens.extend(reserved_tokens)
            special_token_set = set(special_tokens)
            if unknown_token:
                assert unknown_token not in special_token_set, \
                    '`reserved_token` cannot contain `unknown_token`.'
            assert len(special_token_set) == len(special_tokens), \
                '`reserved_tokens` cannot contain duplicate reserved tokens or ' \
                'other special tokens.'
        self._index_special_tokens(unknown_token, special_tokens)

        if counter:
            self._index_counter_keys(counter, unknown_token, special_tokens, max_size, min_freq)

        self._embedding = None

    def _index_special_tokens(self, unknown_token, special_tokens):
        """Indexes unknown and reserved tokens."""
        self._idx_to_token = [unknown_token] if unknown_token else []

        if not special_tokens:
            self._reserved_tokens = None
        else:
            self._reserved_tokens = special_tokens[:]
            self._idx_to_token.extend(special_tokens)

        if unknown_token:
            self._token_to_idx = DefaultLookupDict(C.UNK_IDX)
        else:
            self._token_to_idx = {}
        self._token_to_idx.update((token, idx) for idx, token in enumerate(self._idx_to_token))

    def _index_counter_keys(self, counter, unknown_token, special_tokens, max_size,
                            min_freq):
        """Indexes keys of `counter`.


        Indexes keys of `counter` according to frequency thresholds such as `max_size` and
        `min_freq`.
        """

        unknown_and_special_tokens = set(special_tokens) if special_tokens else set()

        if unknown_token:
            unknown_and_special_tokens.add(unknown_token)

        token_freqs = sorted(counter.items(), key=lambda x: x[0])
        token_freqs.sort(key=lambda x: x[1], reverse=True)

        token_cap = len(unknown_and_special_tokens) + (
            len(counter) if not max_size else max_size)

        for token, freq in token_freqs:
            if freq < min_freq or len(self._idx_to_token) == token_cap:
                break
            if token not in unknown_and_special_tokens:
                self._idx_to_token.append(token)
                self._token_to_idx[token] = len(self._idx_to_token) - 1

    @property
    def embedding(self):
        return self._embedding

    @property
    def idx_to_token(self):
        return self._idx_to_token

    @property
    def reserved_tokens(self):
        return self._reserved_tokens

    @property
    def token_to_idx(self):
        return self._token_to_idx

    @property
    def unknown_token(self):
        return self._unknown_token

    @property
    def padding_token(self):
        return self._padding_token

    @property
    def bos_token(self):
        return self._bos_token

    @property
    def eos_token(self):
        return self._eos_token

    def __contains__(self, token):
        """Checks whether a text token exists in the vocabulary.


        Parameters
        ----------
        token : str
            A text token.


        Returns
        -------
        bool
            Whether the text token exists in the vocabulary (including `unknown_token`).
        """

        return token in self._token_to_idx

    def __getitem__(self, tokens):
        """Looks up indices of text tokens according to the vocabulary.

        If `unknown_token` of the vocabulary is None, looking up unknown tokens results in KeyError.

        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.


        Returns
        -------
        int or list of ints
            A token index or a list of token indices according to the vocabulary.
        """

        if not isinstance(tokens, (list, tuple)):
            return self._token_to_idx[tokens]
        else:
            return [self._token_to_idx[token] for token in tokens]

    def __len__(self):
        return len(self._idx_to_token)

    def set_embedding(self, *embeddings):
        """Attaches one or more embeddings to the indexed text tokens.


        Parameters
        ----------
        embeddings : None or tuple of :class:`gluonnlp.embedding.TokenEmbedding` instances
            The embedding to be attached to the indexed tokens. If a tuple of multiple embeddings
            are provided, their embedding vectors will be concatenated for the same token.
        """

        if len(embeddings) == 1 and embeddings[0] is None:
            self._embedding = None
            return

        for embs in embeddings:
            assert isinstance(embs, emb.TokenEmbedding), \
                'The argument `embeddings` must be an instance or a list of instances of ' \
                '`gluonnlp.embedding.TokenEmbedding`.'

        new_embedding = emb.TokenEmbedding(self.unknown_token)
        new_embedding._token_to_idx = self.token_to_idx
        new_embedding._idx_to_token = self.idx_to_token

        new_vec_len = sum(embs.idx_to_vec.shape[1] for embs in embeddings
                          if embs and embs.idx_to_vec is not None)
        new_idx_to_vec = nd.zeros(shape=(len(self), new_vec_len))

        col_start = 0
        # Concatenate all the embedding vectors in embedding.
        for embs in embeddings:
            if embs and embs.idx_to_vec is not None:
                col_end = col_start + embs.idx_to_vec.shape[1]
                # Cancatenate vectors of the unknown token.
                new_idx_to_vec[0, col_start:col_end] = embs[0]
                new_idx_to_vec[1:, col_start:col_end] = embs[self._idx_to_token[1:]]
                col_start = col_end

        new_embedding._idx_to_vec = new_idx_to_vec
        self._embedding = new_embedding

    def to_tokens(self, indices):
        """Converts token indices to tokens according to the vocabulary.


        Parameters
        ----------
        indices : int or list of ints
            A source token index or token indices to be converted.


        Returns
        -------
        str or list of strs
            A token or a list of tokens according to the vocabulary.
        """

        to_reduce = False
        if not isinstance(indices, (list, tuple)):
            indices = [indices]
            to_reduce = True

        max_idx = len(self._idx_to_token) - 1

        tokens = []
        for idx in indices:
            if not isinstance(idx, int) or idx > max_idx:
                raise ValueError('Token index {} in the provided `indices` is invalid.'.format(idx))
            else:
                tokens.append(self._idx_to_token[idx])

        return tokens[0] if to_reduce else tokens

    def to_indices(self, tokens):
        """Looks up indices of text tokens according to the vocabulary.


        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.


        Returns
        -------
        int or list of ints
            A token index or a list of token indices according to the vocabulary.
        """

        return self[tokens]

    def __call__(self, tokens):
        """Looks up indices of text tokens according to the vocabulary.


        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.


        Returns
        -------
        int or list of ints
            A token index or a list of token indices according to the vocabulary.
        """

        return self[tokens]

    def __repr__(self):
        return 'Vocab(size={}, unk="{}", reserved="{}")'.format(len(self), self._unknown_token,
                                                                self._reserved_tokens)

    def to_json(self):
        """Serialize Vocab object to json string.

        This method does not serialize the underlying embedding.
        """
        if self._embedding:
            warnings.warn('Serialization of attached embedding '
                          'to json is not supported. '
                          'You may serialize the embedding to a binary format '
                          'separately using vocab.embedding.serialize')
        vocab_dict = {}
        vocab_dict['idx_to_token'] = self._idx_to_token
        vocab_dict['token_to_idx'] = dict(self._token_to_idx)
        vocab_dict['reserved_tokens'] = self._reserved_tokens
        vocab_dict['unknown_token'] = self._unknown_token
        vocab_dict['padding_token'] = self._padding_token
        vocab_dict['bos_token'] = self._bos_token
        vocab_dict['eos_token'] = self._eos_token
        return json.dumps(vocab_dict)

    @staticmethod
    def from_json(json_str):
        """Deserialize Vocab object from json string.

        Parameters
        ----------
        json_str : str
            Serialized json string of a Vocab object.


        Returns
        -------
        Vocab
        """
        vocab_dict = json.loads(json_str)

        unknown_token = vocab_dict.get('unknown_token')
        vocab = Vocab(unknown_token=unknown_token)
        vocab._idx_to_token = vocab_dict.get('idx_to_token')
        vocab._token_to_idx = vocab_dict.get('token_to_idx')
        if unknown_token:
            vocab._token_to_idx = DefaultLookupDict(vocab._token_to_idx[unknown_token],
                                                    vocab._token_to_idx)
        vocab._reserved_tokens = vocab_dict.get('reserved_tokens')
        vocab._padding_token = vocab_dict.get('padding_token')
        vocab._bos_token = vocab_dict.get('bos_token')
        vocab._eos_token = vocab_dict.get('eos_token')
        return vocab

class UnicodeCharsVocabulary(Vocab):
    """Vocabulary containing character-level and word level information.

    Has a word vocabulary that is used to lookup word ids and
    a character id that is used to map words to arrays of character ids.

    The character ids are defined by ord(c) for c in word.encode('utf-8')
    This limits the total number of possible char ids to 256.
    To this we add 5 additional special ids: begin sentence, end sentence,
        begin word, end word and padding.
    """
    def __init__(self, counter=None, max_word_length=50, max_size=None, min_freq=1, unknown_token='<unk>',
                 padding_token='<pad>', bos_token='<bos>', eos_token='<eos>', reserved_tokens=None):
        super(UnicodeCharsVocabulary, self).__init__(counter, max_size, min_freq, unknown_token, padding_token,
                                                     bos_token, eos_token, reserved_tokens)
        self._max_word_length = max_word_length

        # char ids 0-255 come from utf-8 encoding bytes
        # assign 256-300 to special chars
        self.bos_char = 256  # <begin sentence>
        self.eos_char = 257  # <end sentence>
        self.bow_char = 258  # <begin word>
        self.eow_char = 259  # <end word>
        self.pad_char = 260 # <padding>

        if counter:
            self.num_words = self.__len__()

            self._word_char_ids = np.zeros([self.num_words, max_word_length],
                dtype=np.int32)

            # the charcter representation of the begin/end of sentence characters
            def _make_bos_eos(c):
                r = np.zeros([self.max_word_length], dtype=np.int32)
                r[:] = self.pad_char
                r[0] = self.bow_char
                r[1] = c
                r[2] = self.eow_char
                return r
            self.bos_chars = _make_bos_eos(self.bos_char)
            self.eos_chars = _make_bos_eos(self.eos_char)

            for i, word in enumerate(self._token_to_idx):
                self._word_char_ids[i] = self._convert_word_to_char_ids(word)

            self._word_char_ids[self._token_to_idx[self.bos_token]] = self.bos_chars
            self._word_char_ids[self._token_to_idx[self.eos_token]] = self.eos_chars

    @property
    def word_char_ids(self):
        return self._word_char_ids

    @property
    def size(self):
        return self.num_words

    @property
    def max_word_length(self):
        return self._max_word_length

    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_word_length], dtype=np.int32)
        code[:] = self.pad_char

        word_encoded = word.encode('utf-8', 'ignore')[:(self.max_word_length-2)]
        code[0] = self.bow_char
        for k, chr_id in enumerate(word_encoded, start=1):
            code[k] = chr_id
        code[k + 1] = self.eow_char

        return code

    def word_to_char_ids(self, word):
        if word in self._token_to_idx:
            return self._word_char_ids[self._token_to_idx[word]]
        else:
            return self._convert_word_to_char_ids(word)

    def array_to_char_ids(self, input_array, max_word_length):
        char_array = nd.full((input_array.shape[0], input_array.shape[1], max_word_length), self.pad_char)

        for i in range(input_array.shape[0]):
            for j in range(input_array.shape[1]):
                word = input_array[i][j]
                if word in self._token_to_idx:
                    char_array[i][j] = self._word_char_ids[self._token_to_idx[word]]
                else:
                    word_encoded = word.encode('utf-8', 'ignore')[:(self.max_word_length - 2)]
                    char_array[i][j][0] = self.bow_char
                    for k, chr_id in enumerate(word_encoded, start=1):
                        char_array[i][j][k] = chr_id
                    char_array[i][j][k + 1] = self.eow_char

        char_array += 1
        return char_array

    def dataset_to_char_ids(self, dataset, batch_size, sample_len, max_word_length):
        char_dataset = nd.full((batch_size, sample_len, max_word_length), self.pad_char)

        for i, word in enumerate(dataset):
            if word in self._token_to_idx:
                char_dataset[i // sample_len][i % sample_len] = self._word_char_ids[self._token_to_idx[word]]
            else:
                word_encoded = word.encode('utf-8', 'ignore')[:(self.max_word_length - 2)]
                char_dataset[i // sample_len][i % sample_len][0] = self.bow_char
                for k, chr_id in enumerate(word_encoded, start=1):
                    char_dataset[i // sample_len][i % sample_len][k] = chr_id
                char_dataset[i // sample_len][i % sample_len][k + 1] = self.eow_char

        char_dataset += 1

        return char_dataset

    def encode_chars(self, sentence, reverse=False, split=True):
        '''
        Encode the sentence as a white space delimited string of tokens.
        '''
        if split:
            chars_ids = [self.word_to_char_ids(cur_word)
                     for cur_word in sentence.split()]
        else:
            chars_ids = [self.word_to_char_ids(cur_word)
                     for cur_word in sentence]
        if reverse:
            return np.vstack([self.eos_chars] + chars_ids + [self.bos_chars])
        else:
            return np.vstack([self.bos_chars] + chars_ids + [self.eos_chars])

    def __getitem__(self, tokens):
        """Looks up indices of text tokens according to the vocabulary.

        If `unknown_token` of the vocabulary is None, looking up unknown tokens results in KeyError.

        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.


        Returns
        -------
        int or list of ints
            A token index or a list of token indices according to the vocabulary.
        """

        if isinstance(tokens, (list, tuple)):
            return [self._token_to_idx[token] for token in tokens]
        elif isinstance(tokens, np.ndarray):
            vfunc = np.vectorize(self._token_to_idx.__getitem__)
            return vfunc(tokens)
        else:
            return self._token_to_idx[tokens]