'''Author: Brandon Trabucco, Copyright 2019
Implements an original idea from my research with Dr. Jean Oh at CMU.
np preprint yet.'''


import numpy as np
import tensorflow as tf
import collections
from detailed_captioning.cells.image_caption_cell import ImageCaptionCell


def BestFirstInsert(running_ids, lengths, word_ids, pointer_ids):
    
    def best_first_function(running_ids, lengths, word_ids, pointer_ids):
        """ Inserts into the running_ids list using the best_first approach.
        Args:
            running_ids: a tf.int32 Tensor has shape [batch_size, max_sequence_length]
            lengths: a tf.int32 Tensor has shape [batch_size]
            word_ids: a tf.int32 Tensor has shape [batch_size]
            pointer_ids: a tf.int32 Tensor has shape [batch_size]
        Returns:
            running_ids: a tf.int32 Tensor has shape [batch_size, max_sequence_length] 
            lengths: a tf.int32 Tensor has shape [batch_size] """
        
        out_running_ids = np.zeros(running_ids.shape, dtype=np.int32)
        out_lengths = np.zeros(lengths.shape, dtype=np.int32)
        for i in range(running_ids.shape[0]):
            if pointer_ids[i] >= 0 and pointer_ids[i] < running_ids.shape[1]:
                (out_running_ids[i, :pointer_ids[i] + 1], out_running_ids[i, pointer_ids[i] + 1], 
                 out_running_ids[i, pointer_ids[i] + 2:], out_lengths[i]) = (
                    running_ids[i, :pointer_ids[i] + 1], word_ids[i], 
                    running_ids[i, pointer_ids[i] + 1:-1], lengths[i] + 1)
            else:
                out_running_ids[i, :], out_lengths[i] = running_ids[i, :], lengths[i]
                
        return out_running_ids, out_lengths
    
    return tf.py_func(best_first_function, [running_ids, lengths, word_ids, 
        pointer_ids], tf.int32)


_BestFirstStateTuple = collections.namedtuple("BestFirstStateTuple", (
    "running_ids", "lengths", "pointer_ids"))
class BestFirstStateTuple(_UpDownStateTuple):
    __slots__ = ()
    @property
    def dtype(self):
        (running_ids, lengths, pointer_ids) = self
        if (not running_ids.dtype == tf.int32 or 
                not lengths.dtype == tf.int32 or
                not pointer_ids.dtype == tf.int32):
            raise TypeError("Inconsistent internal state: %s vs %s vs %s" %
                            (str(running_ids.dtype), 
                             str(lengths.dtype), 
                             str(pointer_ids.dtype)))
        return tf.int32

    
class BestFirstCell(ImageCaptionCell):

    def __init__(self, image_caption_cell, word_embeddings,
                 maximum_sequence_length=1000,
                 name="best_first", initializer=None, **kwargs):
        super(BestFirstCell, self).__init__(**kwargs)
        self.image_caption_cell = image_caption_cell
        self.embeddings_map = tf.get_variable(name + "/embeddings_map", dtype=tf.float32,
            initializer=tf.constant(word_embeddings, dtype=tf.float32))
        self.pointer_layer = tf.layers.Dense(1, kernel_initializer=initializer, 
            name=(name + "/pointer_layer"))
        self._state_size = BestFirstStateTuple(running_ids=maximum_sequence_length, 
            lengths=1, pointer_ids=1)
        self._output_size = self.image_caption_cell.output_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state):
        
        previous_running_ids = state.running_ids
        previous_lengths = tf.squeeze(state.lengths)
        previous_pointer_ids = tf.squeeze(state.pointer_ids)
        previous_word_ids = inputs
        current_running_ids, current_lengths = BestFirstInsert(previous_running_ids, 
            previous_lengths, previous_word_ids, previous_pointer_ids)
        batch_size = tf.shape(current_running_ids)[0]
        initial_state = self.image_caption_cell.zero_state(batch_size, tf.float32)
        activations, _state = tf.nn.dynamic_rnn(self.image_caption_cell, 
            tf.nn.embedding_lookup(self.embeddings_map, current_running_ids),
            sequence_length=current_lengths, initial_state=initial_state)
        # TODO: Need to mask out the out of bounds logits based on length
        pointer_logits = tf.squeeze(self.pointer_layer(activations))
        # TODO: This will need to change and use the ground truth
        current_pointer_ids = tf.argmax(pointer_logits, axis=-1, output_type=tf.int32)
        next_state = BestFirstStateTuple(running_ids=current_running_ids, 
            lengths=tf.expand_dims(current_lengths, 1), 
            pointer_ids=tf.expand_dims(current_pointer_ids, 1))
        pointed_activations = tf.gather(activations, current_pointer_ids, axis=1)
        return pointed_activations, next_state
    
    @property
    def trainable_variables(self):
        cell_variables = (self.language_lstm.trainable_variables + 
            self.pointer_layer.trainable_variables + [self.embeddings_map])
        return cell_variables
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        cell_variables = (self.language_lstm.variables + 
            self.pointer_layer.variables + [self.embeddings_map])
        return cell_variables
    
    @property
    def weights(self):
        return self.variables
    
