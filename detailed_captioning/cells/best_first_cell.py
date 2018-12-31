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
                (out_running_ids[i, :pointer_ids[i]], out_running_ids[i, pointer_ids[i]], 
                 out_running_ids[i, pointer_ids[i] + 1:], out_lengths[i]) = (
                    running_ids[i, :pointer_ids[i]], word_ids[i], 
                    running_ids[i, pointer_ids[i]:-1], lengths[i] + 1)
            else:
                out_running_ids[i, :], out_lengths[i] = running_ids[i, :], lengths[i]
                
        return out_running_ids, out_lengths
    
    y = tf.py_func(best_first_function, [running_ids, lengths, word_ids, 
        pointer_ids], [tf.int32, tf.int32])
    batch_size = running_ids.get_shape()[0]
    max_sequence_length = running_ids.get_shape()[1]
    y[0].set_shape([batch_size, max_sequence_length])
    y[1].set_shape([batch_size])
    return y[0], y[1]


def ThermometerEncoding(ids, capacity):
    
    def thermometer_encoding_function(ids, capacity):
        """ Looks up a thermometer encoding.
        Args:
            ids: a tf.int32 Tensor has shape [batch_size]
            capacity: a tf.int32 Tensor has shape []
        Returns:
            encoding: a tf.int32 Tensor has shape [batch_size, capacity] """
        
        out_encoding = np.zeros([ids.shape[0], capacity], dtype=np.float32)
        for i in range(ids.shape[0]):
            if ids[i] >= 0 and ids[i] < capacity:
                out_encoding[i, :ids[i]] = np.ones([ids[i]], dtype=np.float32)
                
        return out_encoding
    
    y = tf.py_func(thermometer_encoding_function, [ids, capacity], [tf.float32])
    batch_size = ids.get_shape()[0]
    y[0].set_shape([batch_size, capacity])
    return y[0]


_BestFirstStateTuple = collections.namedtuple("BestFirstStateTuple", ("sentinel",
    "caption", "running_words", "running_pointers", "lengths", "previous_pointer"))
class BestFirstStateTuple(_BestFirstStateTuple):
    __slots__ = ()
    @property
    def dtype(self):
        (sentinel, caption, running_words, running_pointers, lengths, 
         previous_pointer) = self
        if (not sentinel.dtype == tf.float32 or
                not caption.dtype == tf.int32 or 
                not running_words.dtype == tf.int32 or
                not running_pointers.dtype == tf.int32 or
                not lengths.dtype == tf.int32 or
                not previous_pointer.dtype == tf.int32):
            raise TypeError("Inconsistent internal state: %s vs %s vs %s vs %s vs %s vs %s" %
                            (str(sentinel.dtype), 
                             str(caption.dtype), 
                             str(running_words.dtype), 
                             str(running_pointers.dtype), 
                             str(lengths.dtype), 
                             str(previous_pointer.dtype)))
        return sentinel.dtype # sentinel is a workaround for seq2seq/beam_search_decoder.py

    
class BestFirstCell(ImageCaptionCell):

    def __init__(self, image_caption_cell, embeddings_map,
                 maximum_sequence_length=1000,
                 name="best_first", initializer=None, **kwargs):
        super(BestFirstCell, self).__init__(**kwargs)
        self.image_caption_cell = image_caption_cell
        self.embeddings_map = embeddings_map
        self.maximum_sequence_length = maximum_sequence_length
        self.pointer_layer = tf.layers.Dense(1, kernel_initializer=initializer, 
            name=(name + "/pointer_layer"))
        self._state_size = BestFirstStateTuple(
            sentinel=1,
            caption=maximum_sequence_length, 
            running_words=maximum_sequence_length, 
            running_pointers=maximum_sequence_length, 
            lengths=1, 
            previous_pointer=1)
        self._output_size = self.image_caption_cell.output_size
        self.is_using_dynamic_rnn = False

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return (self._output_size if not self.is_using_dynamic_rnn else 
            self._output_size + self.maximum_sequence_length)
        
    @property
    def mean_image_features(self):
        return self._mean_image_features
    
    @property
    def mean_object_features(self):
        return self._mean_object_features
        
    @property
    def spatial_image_features(self):
        return self._spatial_image_features
    
    @property
    def spatial_object_features(self):
        return self._spatial_object_features
    
    @mean_image_features.setter
    def mean_image_features(self, x):
        self.image_caption_cell.mean_image_features = x
        self._mean_image_features = x
    
    @mean_object_features.setter
    def mean_object_features(self, x):
        self.image_caption_cell.mean_object_features = x
        self._mean_object_features = x
    
    @spatial_image_features.setter
    def spatial_image_features(self, x):
        self.image_caption_cell.spatial_image_features = x
        self._spatial_image_features = x
    
    @spatial_object_features.setter
    def spatial_object_features(self, x):
        self.image_caption_cell.spatial_object_features = x
        self._spatial_object_features = x

    def __call__(self, inputs, state):
        # TODO: need to obtain the pointer ids during inference and logits during training
        # Solution: pass this through using the state
        
        sentinel = state.sentinel
        caption = state.caption
        running_words = state.running_words
        running_pointers = state.running_pointers
        lengths = tf.squeeze(state.lengths)
        previous_pointer = tf.squeeze(state.previous_pointer)
        
        if self.is_using_dynamic_rnn:
            previous_word, next_pointer = inputs[:, 0], inputs[:, 1]
        else:
            previous_word, next_pointer = tf.squeeze(inputs), None
            
        next_caption, next_lengths = BestFirstInsert(caption, lengths, previous_word, 
            previous_pointer)
        next_running_words, _next_lengths = BestFirstInsert(running_words, lengths, 
            previous_word, lengths)
        next_running_pointers, _next_lengths = BestFirstInsert(running_pointers, lengths, 
            previous_pointer, lengths)
        
        batch_size = tf.shape(inputs)[0]
        initial_state = self.image_caption_cell.zero_state(batch_size, tf.float32)
        activations, _state = tf.nn.dynamic_rnn(self.image_caption_cell, 
            tf.nn.embedding_lookup(self.embeddings_map, next_caption),
            sequence_length=next_lengths, initial_state=initial_state)
        
        indicator = ThermometerEncoding(next_lengths, self.maximum_sequence_length)
        pointer_logits = tf.squeeze(self.pointer_layer(activations), 2)
        if next_pointer is None:
            next_pointer = tf.argmax(
                tf.nn.softmax(pointer_logits) * indicator, axis=1, output_type=tf.int32)
            
        next_state = BestFirstStateTuple(
            sentinel=sentinel,
            caption=next_caption, 
            running_words=next_running_words, 
            running_pointers=next_running_pointers,
            lengths=tf.expand_dims(next_lengths, 1),
            previous_pointer=tf.expand_dims(next_pointer, 1))
        
        pointed_activations = tf.squeeze(tf.gather_nd(activations, tf.concat([
            tf.expand_dims(tf.range(batch_size), 1), tf.expand_dims(next_pointer, 1)], 1)))
        
        if self.is_using_dynamic_rnn:
            # Workaround so that we can train the pointer logits using cross entropy
            pointed_activations = tf.concat([pointed_activations, pointer_logits], 1)
                    
        return pointed_activations, next_state 
    
    @property
    def trainable_variables(self):
        # The embeddings are already saved inside the best first captioner
        cell_variables = (self.image_caption_cell.trainable_variables + 
            self.pointer_layer.trainable_variables)
        return cell_variables
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        # The embeddings are already saved inside the best first captioner
        cell_variables = (self.image_caption_cell.variables + 
            self.pointer_layer.variables)
        return cell_variables
    
    @property
    def weights(self):
        return self.variables
    
