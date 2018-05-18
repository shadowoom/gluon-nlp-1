
import argparse

import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset, LMDataset


def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file)

    # define the options
    batch_size = 20  # batch size for each GPU
    n_gpus = 1

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = 2088628

    options = {
     'bidirectional': True,
     # 'char_cnn': {'activation': 'relu',
     #  'embedding': {'dim': 16},
     #  'filters': [[1, 32],
     #   [2, 32],
     #   [3, 64],
     #   [4, 128],
     #   [5, 256],
     #   [6, 512],
     #   [7, 1024]],
     #  'max_characters_per_token': 50,
     #  'n_characters': 261,
     #  'n_highway': 2},

     'dropout': 0.5,
     'learning_rate': 0.002,

     'lstm': {
      'cell_clip': None,
      'dim': 650,
      'n_layers': 2,
      'proj_clip': None,
      'projection_dim': 650,
      'use_skip_connections': False},

     'all_clip_norm_val': 0.25,

     'n_epochs': 1,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 35,
     'share_embedding_softmax': True,
     'sample_softmax': False,
     # 'n_negative_samples_batch': 8192,
    }

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=True)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    if args.load_latest:
        options, ckpt_file = load_options_latest_checkpoint(args.save_dir)
    train(options, data, n_gpus, tf_save_dir, tf_log_dir, restart_ckpt_file=ckpt_file if args.load_latest else args.load)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--load', help='Checkpoint location')
    parser.add_argument('--load_latest', action='store_true')

    args = parser.parse_args()
    main(args)


