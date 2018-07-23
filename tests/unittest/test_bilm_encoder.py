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
