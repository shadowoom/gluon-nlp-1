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

from __future__ import print_function

import sys
import numpy
import json
import os
import h5py
import io

import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
import pytest
import mxnet as mx

from gluonnlp.model import BiLMEncoder


def test_bilmencoder():
    encoder = BiLMEncoder(mode='lstmpc', num_layers=1, input_size=10,
                       hidden_size=10, dropout=0.1, skip_connection=True,
                       proj_size=10, cell_clip=1, proj_clip=1)
    encoder.initialize()
    inputs0 = mx.random.uniform(shape=(20, 5, 10))
    inputs1 = mx.random.uniform(shape=(20, 5, 10))
    inputs = (inputs0, inputs1)
    states0 = []
    states00 = mx.random.uniform(shape=(5, 10))
    states01 = mx.random.uniform(shape=(5, 10))
    states0.append(states00)
    states0.append(states01)
    states_forward = []
    states_forward.append(states0)
    states1 = []
    states10 = mx.random.uniform(shape=(5, 10))
    states11 = mx.random.uniform(shape=(5, 10))
    states1.append(states10)
    states1.append(states11)
    states_backward = []
    states_backward.append(states1)
    states = []
    states.append(states_forward)
    states.append(states_backward)
    outputs, out_states = encoder(inputs, states)
    assert outputs[0][0].shape == (20, 5, 10), outputs[0][0].shape
    assert outputs[1][0].shape == (20, 5, 10), outputs[1][0].shape
    assert len(outputs) == 2, len(outputs)
    assert len(out_states) == 2, len(out_states)
    assert out_states[0][0][0].shape == (5, 10), out_states[0][0][0].shape
    assert out_states[0][0][1].shape == (5, 10), out_states[0][0][1].shape
    assert out_states[1][0][0].shape == (5, 10), out_states[0][1][0].shape
    assert out_states[1][0][1].shape == (5, 10), out_states[0][1][1].shape


def test_elmo_bilm():
    import numpy as np
    import gluonnlp.data.batchify as btf

    class SampleBatchify(object):
        """Transform the dataset into N independent sequences, where N is the batch size.
        Parameters
        ----------
        vocab : gluonnlp.Vocab
            The vocabulary to use for numericalizing the dataset. Each token will be mapped to the
            index according to the vocabulary.
        batch_size : int
            The number of samples in each batch.
        """

        def __init__(self, batch_size, char_vocab=None, max_word_length=None):
            # self._vocab = vocab
            self._batch_size = batch_size
            self._char_vocab = char_vocab
            self._max_word_length = max_word_length

        def __call__(self, data):
            """Batchify a dataset.
            Parameters
            ----------
            data : mxnet.gluon.data.Dataset
                A flat dataset to be batchified.
            Returns
            -------
            mxnet.gluon.data.Dataset
                NDArray of shape (len(data) // N, N) where N is the batch_size
                wrapped by a mxnet.gluon.data.SimpleDataset. Excessive tokens that
                don't align along the batches are discarded.
            """
            sample_len = len(data) // self._batch_size
            # word_nd = mx.nd.array(
            #     self._vocab[data[:sample_len * self._batch_size]]).reshape(
            #     self._batch_size, -1).T
            # if self._char_vocab is None:
            #     return word_nd, None
            char_array = []
            for token in data[:sample_len * self._batch_size]:
                print(token)
                if len(token) < self._max_word_length:
                    for c in list(token):
                        char_array.append(self._char_vocab[c])
                    for i in range(self._max_word_length - len(token)):
                        char_array.append(nlp.model.ELMoCharacterMapper.padding_character+1)
                else:
                    for c in list(token)[:self._max_word_length]:
                        char_array.append(self._char_vocab[c])
            char_nd = mx.nd.array(char_array).reshape(sample_len * self._batch_size, -1)
            char_nd_batches = char_nd.split(axis=0, num_outputs=self._batch_size)
            char_batches_reshaped = []
            for i in range(len(char_nd_batches[0])):
                char_batch = []
                for j in range(len(char_nd_batches)):
                    char_batch.append(char_nd_batches[j][i])
                char_nd_batch = mx.nd.concat(*char_batch, dim=0).reshape(-1, self._max_word_length)
                char_batches_reshaped.append(char_nd_batch)
            char_nd_batches_reshaped = mx.nd.concat(*char_batches_reshaped, dim=0) \
                .reshape(sample_len, self._batch_size, self._max_word_length)
            char_nd = char_nd_batches_reshaped
            return char_nd

    class DataListTransform(object):
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

        def __call__(self, sentence):
            # if self._src_max_len > 0:
            #     src_sentence = self._src_vocab[src.split()[:self._src_max_len]]
            # else:
            #     src_sentence = self._src_vocab[src.split()]
            # if self._tgt_max_len > 0:
            #     tgt_sentence = self._tgt_vocab[tgt.split()[:self._tgt_max_len]]
            # else:
            #     tgt_sentence = self._tgt_vocab[tgt.split()]
            sentence_chars = [np.array(self._convert_word_to_char_ids(token)) for token in sentence]
            sentence_chars_nd = np.stack(sentence_chars, axis=0)

            # src_sentence.append(self. [self._src_vocab.eos_token])
            # tgt_sentence.insert(0, self._tgt_vocab[self._tgt_vocab.bos_token])
            # tgt_sentence.append(self._tgt_vocab[self._tgt_vocab.eos_token])
            # src_npy = np.array(src_sentence, dtype=np.int32)
            # tgt_npy = np.array(tgt_sentence, dtype=np.int32)
            return sentence_chars_nd

        def _convert_word_to_char_ids(self, word):
            if word == nlp.model.ELMoCharacterMapper.bos_token:
                char_ids = nlp.model.ELMoCharacterMapper.beginning_of_sentence_characters
            elif word == nlp.model.ELMoCharacterMapper.eos_token:
                char_ids = nlp.model.ELMoCharacterMapper.end_of_sentence_characters
            else:
                word_encoded = word.encode('utf-8', 'ignore')[
                               :(nlp.model.ELMoCharacterMapper.max_word_length - 2)]
                char_ids = [nlp.model.ELMoCharacterMapper.padding_character] \
                           * nlp.model.ELMoCharacterMapper.max_word_length
                char_ids[0] = nlp.model.ELMoCharacterMapper.beginning_of_word_character
                for k, chr_id in enumerate(word_encoded, start=1):
                    char_ids[k] = chr_id
                char_ids[len(word_encoded) + 1] = nlp.model.ELMoCharacterMapper.end_of_word_character

            # +1 one for masking
            return [c + 1 for c in char_ids]

    class SampleDataset(gluon.data.SimpleDataset):
        """Common text dataset that reads a whole corpus based on provided sample splitter
        and word tokenizer.

        The returned dataset includes samples, each of which can either be a list of tokens if tokenizer
        is specified, or otherwise a single string segment produced by the sample_splitter.

        Parameters
        ----------
        filename : str or list of str
            Path to the input text file or list of paths to the input text files.
        encoding : str, default 'utf8'
            File encoding format.
        flatten : bool, default False
            Whether to return all samples as flattened tokens. If True, each sample is a token.
        skip_empty : bool, default True
            Whether to skip the empty samples produced from sample_splitters. If False, `bos` and `eos`
            will be added in empty samples.
        sample_splitter : function, default str.splitlines
            A function that splits the dataset string into samples.
        tokenizer : function or None, default str.split
            A function that splits each sample string into list of tokens. If None, raw samples are
            returned according to `sample_splitter`.
        bos : str or None, default None
            The token to add at the begining of each sequence. If None, or if tokenizer is not
            specified, then nothing is added.
        eos : str or None, default None
            The token to add at the end of each sequence. If None, or if tokenizer is not
            specified, then nothing is added.
        """

        def __init__(self, passages, encoding='utf8', flatten=False, skip_empty=True,
                     sample_splitter=nlp.data.line_splitter, tokenizer=nlp.data.whitespace_splitter,
                     bos=None, eos=None):

            self._passages = passages
            self._encoding = encoding
            self._flatten = flatten
            self._skip_empty = skip_empty
            self._sample_splitter = sample_splitter
            self._tokenizer = tokenizer
            self._bos = bos
            self._eos = eos
            super(SampleDataset, self).__init__(self._read())

        def _read(self):
            all_samples = []
            for passage in self._passages:
                samples = (s.strip() for s in passage)
                if self._tokenizer:
                    samples = [
                        self._corpus_dataset_process(self._tokenizer(s), self._bos, self._eos)
                        for s in samples if s or not self._skip_empty
                    ]
                    if self._flatten:
                        samples = nlp.data.concat_sequence(samples)
                elif self._skip_empty:
                    samples = [s for s in samples if s]

                all_samples += samples
            return all_samples

        def _corpus_dataset_process(self, s, bos, eos):
            tokens = [bos] if bos else []
            tokens.extend(s)
            if eos:
                tokens.append(eos)
            return tokens

    def _load_sentences_embeddings(sentences_json_file, elmo_fixtures_path):
        """
        Load the test sentences and the expected LM embeddings.

        These files loaded in this method were created with a batch-size of 3.
        Due to idiosyncrasies with TensorFlow, the 30 sentences in sentences.json are split into 3 files in which
        the k-th sentence in each is from batch k.

        This method returns a (sentences, embeddings) pair where each is a list of length batch_size.
        Each list contains a sublist with total_sentence_count / batch_size elements.  As with the original files,
        the k-th element in the sublist is in batch k.
        """
        with open(sentences_json_file) as fin:
            sentences = json.load(fin)

        # the expected embeddings
        expected_lm_embeddings = []
        for k in range(len(sentences)):
            embed_fname = os.path.join(elmo_fixtures_path, 'lm_embeddings_{}.hdf5'.format(k)
            )
            expected_lm_embeddings.append([])
            with h5py.File(embed_fname, 'r') as fin:
                for i in range(10):
                    sent_embeds = fin['%s' % i][...]
                    sent_embeds_concat = numpy.concatenate(
                            (sent_embeds[0, :, :], sent_embeds[1, :, :]),
                            axis=-1
                    )
                    expected_lm_embeddings[-1].append(sent_embeds_concat)

        return sentences, expected_lm_embeddings

    def char_splitter(s):
        """Split a string at whitespace.
        Parameters
        ----------
        s : str
            The string to be split
        Returns
        --------
        List[str]
            List of strings. Obtained by calling s.split().
        """
        return list(s)

    def vocab_elmo_transform(vocab):
        _idx_to_token = [None] * (nlp.model.ELMoCharacterMapper.padding_character+2)
        _token_to_idx = {}
        # +1 one for masking
        for index, token in enumerate(vocab._idx_to_token):
            if token == '<unk>' or token == '<pad>' or token in _token_to_idx:
                continue
            if token == nlp.model.ELMoCharacterMapper.bos_token:
                _idx_to_token[nlp.model.ELMoCharacterMapper.beginning_of_sentence_character+1] = token
                _token_to_idx.update({token: nlp.model.ELMoCharacterMapper.beginning_of_sentence_character+1})
            elif token == nlp.model.ELMoCharacterMapper.eos_token:
                _idx_to_token[nlp.model.ELMoCharacterMapper.end_of_sentence_character+1] = token
                _token_to_idx.update({token: nlp.model.ELMoCharacterMapper.end_of_sentence_character+1})
            else:
                token_encoded = token.encode('utf-8', 'ignore')
                for k, chr_id in enumerate(token_encoded, start=1):
                    token_encoded_id = chr_id
                    break
                _idx_to_token[token_encoded_id+1] = token
                _token_to_idx.update({token: token_encoded_id+1})
        vocab._idx_to_token = _idx_to_token
        vocab._token_to_idx = _token_to_idx

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

    def remove_sentence_boundaries(inputs, mask):
        """
        Remove begin/end of sentence embeddings from the batch of sentences.
        Given a batch of sentences with size ``(batch_size, timesteps, dim)``
        this returns a tensor of shape ``(batch_size, timesteps - 2, dim)`` after removing
        the beginning and end sentence markers.  The sentences are assumed to be padded on the right,
        with the beginning of each sentence assumed to occur at index 0 (i.e., ``mask[:, 0]`` is assumed
        to be 1).

        Returns both the new tensor and updated mask.

        This function is the inverse of ``add_sentence_boundary_token_ids``.

        Parameters
        ----------
        tensor : ``torch.Tensor``
            A tensor of shape ``(batch_size, timesteps, dim)``
        mask : ``torch.Tensor``
             A tensor of shape ``(batch_size, timesteps)``

        Returns
        -------
        tensor_without_boundary_tokens : ``torch.Tensor``
            The tensor after removing the boundary tokens of shape ``(batch_size, timesteps - 2, dim)``
        new_mask : ``torch.Tensor``
            The new mask for the tensor of shape ``(batch_size, timesteps - 2)``.
        """
        # TODO: matthewp, profile this transfer
        sequence_lengths = mask.sum(axis=1).asnumpy()
        inputs_shape = list(inputs.shape)
        new_shape = list(inputs_shape)
        new_shape[1] = inputs_shape[1] - 2
        inputs_without_boundary_tokens = mx.nd.zeros(shape=new_shape)
        new_mask = mx.nd.zeros(shape=(new_shape[0], new_shape[1]))
        for i, j in enumerate(sequence_lengths):
            if j > 2:
                inputs_without_boundary_tokens[i, :int((j - 2)), :] = inputs[i, 1:int((j - 1)), :]
                new_mask[i, :int((j - 2))] = 1

        return inputs_without_boundary_tokens, new_mask

    #TODO： 1， make the char id in the vocab, and the rest in the batchify 2, make all in the batchify
    from numpy.testing import assert_almost_equal
    batch_size = 3
    data_dir = '/Users/chgwang/Documents/code/elmo-data/'
    sentences_json_file = 'sentences.json'
    elmo_fixtures_path = './'
    options_file = 'options.json'
    weight_file = 'lm_weights.hdf5'
    sentences, expected_lm_embeddings = _load_sentences_embeddings(data_dir + sentences_json_file, data_dir + elmo_fixtures_path)

    sample_dataset = SampleDataset(sentences, flatten=False)
    sample_dataset = sample_dataset.transform(DataListTransform(), lazy=False)
    sample_dataset = gluon.data.SimpleDataset([(ele, len(ele), i)
                          for i, ele in enumerate(sample_dataset)])
    sample_dataset_batchify_fn = nlp.data.batchify.Tuple(btf.Pad(), btf.Stack(), btf.Stack())
    sample_data_loader = gluon.data.DataLoader(sample_dataset,
                                               batch_size=batch_size,
                                               sampler=mx.gluon.contrib.data.sampler.IntervalSampler(len(sample_dataset), interval=int(len(sample_dataset)/batch_size)),
                                               batchify_fn=sample_dataset_batchify_fn,
                                               num_workers=8)

    elmo_bilm = nlp.model._ElmoBiLm(data_dir + options_file, data_dir + weight_file, cnn_encoder='encoder')
    elmo_bilm.initialize()
    elmo_bilm._token_embedder._load_weights()
    elmo_bilm._token_embedder._char_embedding.weight.set_data(elmo_bilm._token_embedder._char_embedding_weights)
    elmo_bilm._elmo_lstm.load_weights(elmo_bilm._weight_file, elmo_bilm._requires_grad)
    # elmo_bilm.hybridize()
    hidden_state = elmo_bilm._elmo_lstm.begin_state(mx.nd.zeros, batch_size=batch_size)
    for i, batch in enumerate(sample_data_loader):
        print('batch id %d' % i)
        # output, mask = elmo_bilm(batch[0], begin_state)
        # print(output)
        # begin_state = elmo_bilm._elmo_lstm.begin_state(mx.nd.zeros, batch_size=batch_size)
        output, hidden_state, mask = elmo_bilm(batch[0], hidden_state)
        hidden_state = detach(hidden_state)
        top_layer_embeddings, mask = remove_sentence_boundaries(
            output[2],
            mask
        )

        # check the mask lengths
        lengths = mask.asnumpy().sum(axis=1)
        batch_sentences = [sentences[k][i] for k in range(3)]
        expected_lengths = [
            len(sentence.split()) for sentence in batch_sentences
        ]
        # self.assertEqual(lengths.tolist(), expected_lengths)

        # get the expected embeddings and compare!
        expected_top_layer = [expected_lm_embeddings[k][i] for k in range(3)]
        for k in range(3):
            print('k %d' % k)
            # print(top_layer_embeddings[k, :int(lengths[k]), -1:].asnumpy())
            # print(expected_top_layer[k][:,-1:])
            assert_almost_equal(top_layer_embeddings[k, :int(lengths[k]), :].asnumpy(), expected_top_layer[k], decimal=5)
            # print(numpy.allclose(
            #     top_layer_embeddings[k, :int(lengths[k]), :].asnumpy(),
            #     expected_top_layer[k],
            #     atol=1.0e-3
            # ))

test_elmo_bilm()