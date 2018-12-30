'''Author: Brandon Trabucco, Copyright 2019
Implements an original idea from my research with Dr. Jean Oh at CMU.
np preprint yet.'''


import numpy as np
import tensorflow as tf
import collections
from detailed_captioning.utils import tile_with_new_axis
from detailed_captioning.cells.image_caption_cell import ImageCaptionCell


def BestFirstInsert(running_ids, word_ids, pointer_ids, name=None):
    
    def best_first_function(running_ids, word_ids, pointer_ids):
        """ Inserts into the running_ids list using the best_first approach.
        Args:
            running_ids: a tf.int32 Tensor has shape [batch_size, max_sequence_length]
            word_ids: a tf.int32 Tensor has shape [batch_size]
            pointer_ids: a tf.int32 Tensor has shape [batch_size]
        Returns:
            a tf.int32 Tensor has shape [batch_size, max_sequence_length] """
        out_array = np.zeros(running_ids.shape, dtype=np.int32)
        for i in range(word_ids.size):
            (out_array[i, :pointer_ids[i] + 1], out_array[i, pointer_ids[i] + 1], 
             out_array[i, pointer_ids[i] + 2:]) = (
                 running_ids[i, :pointer_ids[i] + 1], word_ids[i], running_ids[i, pointer_ids[i] + 1:-1])
        return out_array
    
    return tf.py_func(best_first_function, [running_ids, word_ids, 
        pointer_ids], tf.int32, name=name)


_BestFirstStateTuple = collections.namedtuple("BestFirstStateTuple", ("running_ids", "length"))
class BestFirstStateTuple(_UpDownStateTuple):
    __slots__ = ()
    @property
    def dtype(self):
        (running_ids, length) = self
        if not running_ids.dtype == length.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                            (str(running_ids.dtype), str(length.dtype)))
        return running_ids.dtype

    
class BestFirstCell(ImageCaptionCell):

    def __init__(self, image_caption_cell
                 name="best_first", initializer=None, **kwargs):
        super(BestFirstCell, self).__init__(**kwargs)
        self.image_caption_cell = image_caption_cell
        def softmax_attention(x):
            x = tf.transpose(x, [0, 2, 1])
            x = tf.nn.softmax(x)
            x = tf.transpose(x, [0, 2, 1])
            return x
        self.attention_layer = tf.layers.Dense(1, kernel_initializer=initializer, 
            name=(name + "/attention_layer"), activation=softmax_attention)
        self._state_size = BestFirstStateTuple(running_ids=tf.TensorShape([2]))
        self._output_size = self.image_caption_cell.output_size

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
    
