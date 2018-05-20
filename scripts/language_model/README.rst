Embeddings from Language Models (ELMo)
--------------------------------------

[1] elmo_dropout_wikitext-2 (Val PPL 96.42 Test PPL 92.39)

.. code-block:: console

   $ python word_language_model.py --gpus 0 --model lstm --emsize 650 --nhid 650 --nlayers 2 --lr 20 --epochs 750 --batch_size 20 --bptt 35 --dropout 0.5 --dropout_h 0 --dropout_i 0 --dropout_e 0 --weight_dropout 0 --tied --wd 0 --alpha 0 --beta 0 --save elmo_lstm_dropout_wikitext-2.params

[2] elmo_dropout_proj_clip_residual_wikitext-2 (Val PPL 125.84 Test PPL 120.82)

.. code-block:: console

   $ python word_language_model.py --gpus 0 --model lstmp --emsize 650 --nhid 650 --nlayers 2 --optimizer adagrad --lr 0.02 --epochs 750 --batch_size 20 --bptt 35 --dropout 0.5 --dropout_h 0 --dropout_i 0 --dropout_e 0 --weight_dropout 0 --wd 0 --alpha 0 --beta 0 --projsize 650 --projclip 3 --cellclip 3 --skip_connection --save elmo_lstm_dropout_proj_clip_residual_wikitext-2.params

.. note::

    Had some convergence issues before (potentially due to learning rate). Might need some hyperparameter tuning and more regularization due to more parameters.

[3] elmo_dropout_proj_clip_residual_char_wikitext-2 - Not Run Properly

.. code-block:: console

   $ python word_language_model.py --gpus 0 --model lstmp --emsize 650 --nhid 650 --nlayers 2 --optimizer adagrad --lr 0.002 --epochs 750 --batch_size 20 --bptt 35 --dropout 0.5 --dropout_h 0 --dropout_i 0 --dropout_e 0 --weight_dropout 0 --wd 0 --alpha 0 --beta 0 --projsize 650 --projclip 3 --cellclip 3 --skip_connection --char_embedding --hdf5_weight_file='' --save elmo_dropout_proj_clip_residual_char_wikitext-2.params

.. note::

    Might have some issues with running. Will update over weekend. Will be last commit so reset to the previous commit if new changes don't work.

TensorFlow Benchmark Repo - `bilm-tf
<https://github.com/SuperLinguini/bilm-tf>`_.

[1] tensorflow_elmo_dropout_wikitext-2 (Val PPL 96.43 Test PPL 92.65)

.. code-block:: console

   $ See tf_word.py in logs directory for details about parameters. To run, replace existing file in directory bin/ in bilm-tf with tf_word.py and rename to word.py. Then run run.sh.

[2] tensorflow_elmo_dropout_proj_clip_residual_wikitext-2 (Val PPL 96.27 Test PPL 92.27)

.. code-block:: console

   $ See tf_word.py in logs directory for details about parameters. You have to change training.py to use projection when the cell size and projection size are the same. Change cell_clip and proj_clip to 3 in tf_word.py. To run, replace existing file in directory bin/ in bilm-tf with tf_word.py and rename to word.py. Then run run.sh.

[3] tensorflow_elmo_dropout_proj_clip_residual_char_wikitext-2 (Val PPL 94.63 Test PPL 90.80)

.. code-block:: console

   $ See tf_char.py in logs directory for details about parameters. To run, replace existing file in directory bin/ in bilm-tf with tf_char.py and rename to char.py. Then run run_char.sh.
