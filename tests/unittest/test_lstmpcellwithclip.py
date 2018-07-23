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

from gluonnlp.model import LSTMPCellWithClip


def test_lstmpcellwithclip():
    cell = LSTMPCellWithClip(hidden_size=10,
                             projection_size=10,
                             cell_clip=1,
                             projection_clip=1,
                             input_size=10)
    cell.initialize()
    inputs = mx.random.uniform(shape=(5, 10))
    states = []
    states0 = mx.random.uniform(shape=(5, 10))
    states1 = mx.random.uniform(shape=(5, 10))
    states.append(states0)
    states.append(states1)
    outputs, out_states = cell(inputs, states)
    assert outputs.shape == (5, 10), outputs.shape
    assert len(out_states) == 2, len(out_states)
    assert out_states[0].shape == (5, 10), out_states[0].shape
    assert out_states[1].shape == (5, 10), out_states[1].shape
