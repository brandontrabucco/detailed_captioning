'''Author: Brandon Trabucco, Copyright 2019
Implements the Show And Tell image caption architecture proposed by
Lu, J. et al. https://arxiv.org/abs/1612.01887'''


import tensorflow as tf
from detailed_captioning.utils import tile_with_new_axis
from detailed_captioning.cells.image_caption_cell import ImageCaptionCell


class VisualSentinelCell(ImageCaptionCell):

    def __init__(self, 
            num_units, use_peepholes=False, cell_clip=None,
            initializer=None, num_proj=None, proj_clip=None,
            num_unit_shards=None, num_proj_shards=None,
            forget_bias=1.0, state_is_tuple=True,
            activation=None, reuse=None, name=None, dtype=None,
            spatial_image_features=None, num_image_features=2048, **kwargs ):
        super(VisualSentinelCell, self).__init__(
            reuse=reuse, name=name, dtype=dtype,
            spatial_image_features=spatial_image_features, **kwargs)
        self.language_lstm = tf.contrib.rnn.LSTMCell(num_units, 
            use_peepholes=use_peepholes, cell_clip=cell_clip,
            initializer=initializer, num_proj=num_proj, proj_clip=proj_clip,
            num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
            forget_bias=forget_bias, state_is_tuple=state_is_tuple,
            activation=activation, reuse=reuse, name=name, dtype=dtype)
        def softmax_attention(x):
            x = tf.transpose(x, [0, 2, 1])
            x = tf.nn.softmax(x)
            x = tf.transpose(x, [0, 2, 1])
            return x
        self.attn_layer = tf.layers.Dense(1, kernel_initializer=initializer, 
            reuse=reuse, name="attention", activation=softmax_attention)
        self.sentinel_gate_layer = tf.layers.Dense(num_units, kernel_initializer=initializer, 
            reuse=reuse, name="sentinel_gate", activation=tf.nn.sigmoid)
        self.sentinel_embeddings_layer = tf.layers.Dense(num_image_features, kernel_initializer=initializer, 
            reuse=reuse, name="sentinel_embeddings", activation=None)
        self._state_size = self.language_lstm.state_size
        self._output_size = self.language_lstm.output_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state):
        l_outputs, l_next_state = self.language_lstm(inputs, state)
        s_inputs = tf.concat([l_next_state.h, inputs], 1)
        sentinel_vector = tf.nn.tanh(l_next_state.c) * self.sentinel_gate_layer(s_inputs)
        sentinel_embeddings = self.sentinel_embeddings_layer(sentinel_vector)
        batch_size = tf.shape(self.spatial_image_features)[0]
        image_height = tf.shape(self.spatial_image_features)[1]
        image_width = tf.shape(self.spatial_image_features)[2]
        image_depth = tf.shape(self.spatial_image_features)[3]
        image_features = tf.reshape(self.spatial_image_features, [batch_size, 
            image_height * image_width, image_depth])
        sentinel_image_features = tf.concat([image_features, tf.expand_dims(sentinel_embeddings, 1)], 1)
        attn_inputs = tf.nn.tanh(tf.concat([ sentinel_image_features, tile_with_new_axis(
            tf.concat(state, 1), [image_height * image_width + 1], [1]) ], 3))
        attended_sif = tf.reduce_sum(sentinel_image_features * self.attn_layer(attn_inputs), [1])
        return attended_sif, l_next_state
    
    @property
    def trainable_variables(self):
        cell_variables = (self.language_lstm.trainable_variables 
                          + self.attn_layer.trainable_variables 
                          + self.sentinel_gate_layer.trainable_variables 
                          + self.sentinel_embeddings_layer.trainable_variables)
        return cell_variables
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        cell_variables = (self.language_lstm.variables 
                          + self.attn_layer.variables
                          + self.sentinel_gate_layer.variables 
                          + self.sentinel_embeddings_layer.variables)
        return cell_variables
    
    @property
    def weights(self):
        return self.variables
    
