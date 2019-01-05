'''Author: Brandon Trabucco, Copyright 2019
Implements a custom visual attention LSTM cell for image captioning.
Anderson, Peter, et al. https://arxiv.org/abs/1707.07998'''


import tensorflow as tf
import collections
from detailed_captioning.utils import tile_with_new_axis
from detailed_captioning.cells.image_caption_cell import ImageCaptionCell


_UpDownStateTuple = collections.namedtuple("UpDownStateTuple", ("visual", "language"))
class UpDownStateTuple(_UpDownStateTuple):
    __slots__ = ()
    @property
    def dtype(self):
        (visual, language) = self
        if not visual.dtype == language.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                            (str(visual.dtype), str(language.dtype)))
        return visual.dtype


class UpDownCell(ImageCaptionCell):

    def __init__(self, 
            num_units, use_peepholes=False, cell_clip=None,
            initializer=None, num_proj=None, proj_clip=None,
            num_unit_shards=None, num_proj_shards=None,
            forget_bias=1.0, state_is_tuple=True,
            activation=None, reuse=None, name="up_down", dtype=None,
            mean_image_features=None, mean_object_features=None, **kwargs ):
        super(UpDownCell, self).__init__(
            reuse=reuse, name=name, dtype=dtype,
            mean_image_features=mean_image_features, 
            mean_object_features=mean_object_features, **kwargs )
        self.visual_lstm = tf.contrib.rnn.LSTMCell(num_units, 
            use_peepholes=use_peepholes, cell_clip=cell_clip,
            initializer=initializer, num_proj=num_proj, proj_clip=proj_clip,
            num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
            forget_bias=forget_bias, state_is_tuple=state_is_tuple,
            activation=activation, reuse=reuse, name=(name + "/visual"), dtype=dtype)
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
        self.attention_layer = tf.layers.Dense(1, kernel_initializer=initializer, 
            name=(name + "/attention_layer"), activation=softmax_attention)
        self._state_size = UpDownStateTuple(
            self.visual_lstm.state_size, self.language_lstm.state_size)
        self._output_size = self.language_lstm.output_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state):
        v_inputs = tf.concat([tf.concat(state.language, 1), self.mean_image_features, inputs], 1)
        v_outputs, v_next_state = self.visual_lstm(v_inputs, state.visual)
        attention_inputs = tf.concat([ self.mean_object_features, tile_with_new_axis(v_outputs, [
            tf.shape(self.mean_object_features)[1]], [1]) ], 2)
        attended_features = tf.reduce_sum(self.mean_object_features * self.attention_layer(
            attention_inputs), 1)
        l_inputs = tf.concat([v_outputs, attended_features], 1)
        l_outputs, l_next_state = self.language_lstm(l_inputs, state.language)
        return l_outputs, UpDownStateTuple(v_next_state, l_next_state)
    
    @property
    def trainable_variables(self):
        cell_variables = (self.visual_lstm.trainable_variables 
                + self.language_lstm.trainable_variables + self.attention_layer.trainable_variables)
        return cell_variables
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        cell_variables = (self.visual_lstm.variables 
                + self.language_lstm.variables + self.attention_layer.variables)
        return cell_variables
    
    @property
    def weights(self):
        return self.variables
    
