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
"""
Adapted from https://github.com/google-research/bert/blob/master/modeling.py
Original author Google

Implemented the following paper:

Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
"""
from tvm import relay
from .init import create_workload
from . import layers
import numpy as np

def get_net(batch_size,
            sequence_length,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            vocab_size,
            type_vocab_size,
            max_position_embeddings,
            **kwargs):
    input_shape = (batch_size, sequence_length)
    input_ids = relay.var("input_ids", shape=input_shape, dtype="int32")
    input_mask = relay.var("input_mask", shape=input_shape, dtype="int32")
    token_type_ids = relay.var("token_type_ids", shape=input_shape, dtype="int32")
    embedding_table = relay.const(np.random.random_sample(size=(vocab_size, hidden_size)), "float32")
    token_type_table = relay.const(np.random.random_sample(size=(type_vocab_size, hidden_size)), "float32")
    full_position_embeddings = relay.const(np.random.random_sample(size=(max_position_embeddings, hidden_size)), "float32")
    layer_norm_gamma = relay.const(np.random.random_sample(size=(hidden_size)), "float32")
    layer_norm_beta = relay.const(np.random.random_sample(size=(hidden_size)), "float32")

    # embeddings
    input_ids = relay.expand_dims(input_ids, -1)
    #flat_input_ids = relay.reshape(input_ids, -1)
    output = relay.gather_nd(embedding_table, input_ids)
    output = relay.reshape(output, [batch_size, sequence_length, hidden_size])
    output = relay.cast(output, "float64")

    flat_token_type_ids = relay.reshape(token_type_ids, -1)
    one_hot_ids = relay.one_hot(flat_token_type_ids, relay.const(1, "int32"), relay.const(0, "int32"), type_vocab_size, -1, "int32")
    token_type_embeddings = relay.nn.dense(one_hot_ids, token_type_table)
    token_type_embeddings = relay.reshape(token_type_embeddings, [batch_size, sequence_length, hidden_size])
    token_type_embeddings = relay.cast(token_type_embeddings, "float64")
    output = relay.add(output, token_type_embeddings)

    position_embeddings = relay.strided_slice(full_position_embeddings, [0, 0], [sequence_length, -1])
    position_embeddings = relay.reshape(position_embeddings, (batch_size, sequence_length, hidden_size))

    output = relay.add(output, position_embeddings)
    output = relay.nn.layer_norm(output, layer_norm_gamma, layer_norm_beta)

    # encoder

    # pooler

    # TODO: remove
    return output

def get_workload(batch_size=1,
                 sequence_length=128,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 type_vocab_size=16,
                 vocab_size=10000,
                 max_position_embeddings=512):
    """Get benchmark workload for BERT

    Parameters
    ----------
    batch_size : int, optional
        The batch size used in the model

    sequence_length : int, optional
        The sequence length used in the model

    hidden_size : int, optional
        Size of encoder and pooler layers

    num_hidden_layers : int, optional
        Number of hidden layers in Transformer

    num_attention_heads : int, optional
        Number of attention heads for each attention layer in Transformer

    intermediate_size : int, optional
        Size of intermediate feed-forward layer in Transformer

    vocab_size : int, optional
        Vocabulary size of input_ids

    type_vocab_size : int, optional
        Vocabulary size of token_type_ids

    Returns
    -------
    mod : tvm.relay.Module
        The relay module that contains a BERT network.

    params : dict of str to NDArray
        The parameters.
    """
    net = get_net(batch_size,
                  sequence_length,
                  hidden_size,
                  num_hidden_layers,
                  num_attention_heads,
                  intermediate_size,
                  type_vocab_size,
                  vocab_size,
                  max_position_embeddings)
    return create_workload(net, inputs=["input_ids", "input_mask", "token_ids"])