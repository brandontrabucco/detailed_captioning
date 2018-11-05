'''Author: Brandon Trabucco, Copyright 2019
Implements a custom visual attention LSTM cell for image captioning.
Anderson, Peter, et al. https://arxiv.org/abs/1707.07998'''


import tensorflow as tf
from tensorflow.python.keras import initializers
import collections


def _softmax(x):
    # x is shaped: [batch, num_boxes, depth]
    x = tf.transpose(x, [0, 2, 1])
    return tf.transpose(tf.nn.softmax(x), [0, 2, 1])


def _sigmoid(x):
    # x is shaped: [batch, num_boxes, depth]
    x_size = tf.to_float(tf.shape(x)[1])
    return tf.nn.sigmoid(x) / x_size


def _tile(x, n, d):
    # expand and tile new dimensions of x
    nd = zip(n, d)
    nd = sorted(nd, key=lambda ab: ab[1])
    n, d = zip(*nd)
    for i in sorted(d):
        x = tf.expand_dims(x, i)
    reverse_d = {val: idx for idx, val in enumerate(d)}
    tiles = [n[reverse_d[i]] if i in d else 1 for i, _s in enumerate(x.shape)]
    return tf.tile(x, tiles)


# Used to store the internal states of each LSTM.
_UpDownStateTuple = collections.namedtuple("UpDownStateTuple", ("v", "l"))
class UpDownStateTuple(_UpDownStateTuple):
    __slots__ = ()
    @property
    def dtype(self):
        (v, l) = self
        if not v.dtype == l.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                            (str(v.dtype), str(l.dtype)))
        return v.dtype


# The wrapper for the up-down mechanism
class UpDownCell(tf.contrib.rnn.LayerRNNCell):
    '''Implements the bottom-up top-down attention mechanism from
    Anderson, Peter, et al. https://arxiv.org/abs/1707.07998'''

    def __init__(self, 
            # The default LSTM parameters
            num_units, use_peepholes=False, cell_clip=None,
            initializer=None, num_proj=None, proj_clip=None,
            num_unit_shards=None, num_proj_shards=None,
            forget_bias=1.0, state_is_tuple=True,
            activation=None, reuse=None, name=None, dtype=None,
            # The extra parameters for the up-down mechanism
            image_features=None, object_features=None,
            attention_method='softmax', **kwargs ):
        
        super(UpDownCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs)
        # Collect the image features to attend to during decoding
        self.mif = image_features
        self.mof = object_features
        # Setup the individual LSTM layers
        self.visual_lstm = tf.contrib.rnn.LSTMCell(num_units, 
            use_peepholes=use_peepholes, cell_clip=cell_clip,
            initializer=initializer, num_proj=num_proj, proj_clip=proj_clip,
            num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
            forget_bias=forget_bias, state_is_tuple=True,
            activation=activation, reuse=reuse, name=name, dtype=dtype)
        self.language_lstm = tf.contrib.rnn.LSTMCell(num_units, 
            use_peepholes=use_peepholes, cell_clip=cell_clip,
            initializer=initializer, num_proj=num_proj, proj_clip=proj_clip,
            num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
            forget_bias=forget_bias, state_is_tuple=True,
            activation=activation, reuse=reuse, name=name, dtype=dtype)
        # The method of attention to use (sum = one, or sum at most one)
        # The attention layer uses the context around a unit.
        self.attn_fn = {'sigmoid': _sigmoid, 'softmax': _softmax}[attention_method]
        self.attn_layer = tf.layers.Conv1D(1, 3, kernel_initializer=initializer, 
            padding="same", activation=self.attn_fn, name="attention")
        self._state_size = UpDownStateTuple(
            self.visual_lstm.state_size, self.language_lstm.state_size)
        self._output_size = self.language_lstm.output_size
        
    @property
    def image_features(self):
        return self.mif
    
    @property
    def object_features(self):
        return self.mof
    
    @image_features.setter
    def image_features(self, x):
        self.mif = x
    
    @object_features.setter
    def object_features(self, x):
        self.mof = x

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state):
        
        # Run a two-layer LSTM cell that attends to ROI features.
        # First run the visual LSTM on the imafe features
        v_inputs = tf.concat([tf.concat(state.v, 1), self.mif, inputs], 1)
        v_outputs, v_next_state = self.visual_lstm(v_inputs, state.v)
        # Then compute the attention map across the ROI features
        attn_inputs = tf.concat([self.mof, _tile(v_outputs, [tf.shape(self.mof)[1]], [1])], 2)
        attended_mof = tf.reduce_sum(self.mof * self.attn_layer(attn_inputs), 1)
        # Last run the language LSTM to predict the next word in sequence
        l_inputs = tf.concat([v_outputs, attended_mof, inputs], 1)
        l_outputs, l_next_state = self.language_lstm(l_inputs, state.l)
        return l_outputs, UpDownStateTuple(v_next_state, l_next_state)
    
    @property
    def trainable_variables(self):
        # Handles to the models internal shared trainable variables
        cell_variables = (self.visual_lstm.trainable_variables + self.language_lstm.trainable_variables 
                + self.attn_layer.trainable_variables)
        return cell_variables
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        # Handles to the models internal shared trainable variables
        cell_variables = (self.visual_lstm.variables + self.language_lstm.variables 
                + self.attn_layer.variables)
        return cell_variables
    
    @property
    def weights(self):
        return self.variables
    
