'''Author: Brandon Trabucco, Copyright 2019
Implements the Show And Tell image caption architecture proposed by
Vinyals, O. et al. https://arxiv.org/abs/1411.4555.'''


import tensorflow as tf
from detailed_captioning.cells.image_caption_cell import ImageCaptionCell


class ShowAndTellCell(ImageCaptionCell):

    def __init__(self, 
            num_units, use_peepholes=False, cell_clip=None,
            initializer=None, num_proj=None, proj_clip=None,
            num_unit_shards=None, num_proj_shards=None,
            forget_bias=1.0, state_is_tuple=True,
            activation=None, reuse=None, name=None, dtype=None,
            mean_image_features=None, **kwargs ):
        super(ShowAndTellCell, self).__init__(
            reuse=reuse, name=name, dtype=dtype,
            mean_image_features=mean_image_features, **kwargs)
        self.language_lstm = tf.contrib.rnn.LSTMCell(num_units, 
            use_peepholes=use_peepholes, cell_clip=cell_clip,
            initializer=initializer, num_proj=num_proj, proj_clip=proj_clip,
            num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
            forget_bias=forget_bias, state_is_tuple=state_is_tuple,
            activation=activation, reuse=reuse, name=name, dtype=dtype)
        self._state_size = self.language_lstm.state_size
        self._output_size = self.language_lstm.output_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state):
        l_inputs = tf.concat([self.mean_image_features, inputs], 1)
        l_outputs, l_next_state = self.language_lstm(l_inputs, state)
        return l_outputs, l_next_state
    
    @property
    def trainable_variables(self):
        cell_variables = self.language_lstm.trainable_variables 
        return cell_variables
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        cell_variables = self.language_lstm.variables 
        return cell_variables
    
    @property
    def weights(self):
        return self.variables
    
