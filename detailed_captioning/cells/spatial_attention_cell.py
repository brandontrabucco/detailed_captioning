'''Author: Brandon Trabucco, Copyright 2019
Implements the Spatial Attention post-RNN mechanism proposed in  
Lu, J. et al. https://arxiv.org/abs/1612.01887'''


import tensorflow as tf
from detailed_captioning.utils import tile_with_new_axis
from detailed_captioning.utils import collapse_dims
from detailed_captioning.cells.image_caption_cell import ImageCaptionCell


class SpatialAttentionCell(ImageCaptionCell):

    def __init__(self, 
            num_units, use_peepholes=False, cell_clip=None,
            initializer=None, num_proj=None, proj_clip=None,
            num_unit_shards=None, num_proj_shards=None,
            forget_bias=1.0, state_is_tuple=True,
            activation=None, reuse=None, name="spatial_attention", dtype=None,
            spatial_image_features=None, num_image_features=2048, **kwargs ):
        super(SpatialAttentionCell, self).__init__(
            reuse=reuse, name=name, dtype=dtype,
            spatial_image_features=spatial_image_features, **kwargs)
        self.language_lstm = tf.contrib.rnn.LSTMCell(num_units, 
            use_peepholes=use_peepholes, cell_clip=cell_clip,
            initializer=initializer, num_proj=num_proj, proj_clip=proj_clip,
            num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
            forget_bias=forget_bias, state_is_tuple=state_is_tuple,
            activation=activation, reuse=reuse, name=(name + "/language"), dtype=dtype)
        def softmax_attention(x):
            x = tf.transpose(x, [0, 2, 1])
            x = tf.nn.softmax(x)
            x = tf.transpose(x, [0, 2, 1])
            return x
        self.attn_layer = tf.layers.Dense(1, kernel_initializer=initializer, 
            name=(name + "/attention_layer"), activation=softmax_attention)
        self._state_size = self.language_lstm.state_size
        self._output_size = self.language_lstm.output_size + num_image_features

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state):
        l_inputs = tf.concat([tf.reduce_mean(self.spatial_image_features, [1, 2]), inputs], 1)
        l_outputs, l_next_state = self.language_lstm(l_inputs, state)
        s_inputs = tf.concat([l_outputs, inputs], 1)
        image_height = tf.shape(self.spatial_image_features)[1]
        image_width = tf.shape(self.spatial_image_features)[2]
        image_features = collapse_dims(self.spatial_image_features, [1, 2])
        attn_inputs = tf.concat([ image_features, tile_with_new_axis(l_outputs, [
            image_height * image_width], [1]) ], 2)
        attended_features = tf.reduce_sum(image_features * self.attn_layer(attn_inputs), [1])
        return tf.concat([attended_features, l_outputs], 1), l_next_state
    
    @property
    def trainable_variables(self):
        cell_variables = (self.language_lstm.trainable_variables 
                          + self.attn_layer.trainable_variables)
        return cell_variables
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        cell_variables = (self.language_lstm.variables 
                          + self.attn_layer.variables)
        return cell_variables
    
    @property
    def weights(self):
        return self.variables
    
