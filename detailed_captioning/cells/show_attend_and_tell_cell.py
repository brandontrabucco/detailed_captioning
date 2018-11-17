'''Author: Brandon Trabucco, Copyright 2019
Implements the Show And Tell image caption architecture proposed by
Vinyals, O. et al. https://arxiv.org/abs/1411.4555.'''


import tensorflow as tf
from detailed_captioning.utils import tile_with_new_axis
from detailed_captioning.cells.image_caption_cell import ImageCaptionCell


class ShowAttendAndTellCell(ImageCaptionCell):

    def __init__(self, 
            num_units, use_peepholes=False, cell_clip=None,
            initializer=None, num_proj=None, proj_clip=None,
            num_unit_shards=None, num_proj_shards=None,
            forget_bias=1.0, state_is_tuple=True,
            activation=None, reuse=None, name=None, dtype=None,
            spatial_image_features=None, **kwargs ):
        super(ShowAttendAndTellCell, self).__init__(
            reuse=reuse, name=name, dtype=dtype,
            spatial_image_features=spatial_image_features, **kwargs)
        self.language_lstm = tf.contrib.rnn.LSTMCell(num_units, 
            use_peepholes=use_peepholes, cell_clip=cell_clip,
            initializer=initializer, num_proj=num_proj, proj_clip=proj_clip,
            num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
            forget_bias=forget_bias, state_is_tuple=state_is_tuple,
            activation=activation, reuse=reuse, name=name, dtype=dtype)
        self.attn_layer = tf.layers.Conv1D(1, 3, kernel_initializer=initializer, 
            padding="same", name="attention",
            activation=lambda x: tf.transpose(tf.nn.softmax(tf.transpose(x, [0, 2, 1])), [0, 2, 1]))
        self._state_size = self.language_lstm.state_size
        self._output_size = self.language_lstm.output_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state):
        batch_size = tf.shape(self.spatial_image_features)[0]
        image_height = tf.shape(self.spatial_image_features)[1]
        image_width = tf.shape(self.spatial_image_features)[2]
        image_depth = tf.shape(self.spatial_image_features)[3]
        flat_image_features = tf.reshape(self.spatial_image_features, [batch_size, 
            image_height * image_width, image_depth])
        attn_inputs = tf.concat([ flat_image_features, tile_with_new_axis(
            tf.concat(state, 1), [image_height * image_width], [1]) ], 2)
        attended_sif = tf.reduce_sum(flat_image_features * self.attn_layer(attn_inputs), 1)
        l_inputs = tf.concat([attended_sif, inputs], 1)
        l_outputs, l_next_state = self.language_lstm(l_inputs, state)
        return l_outputs, l_next_state
    
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
    
