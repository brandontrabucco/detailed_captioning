"""Author: Brandon Trabucco, Copyright 2019
Implements the Best First Module for image captioning."""


import tensorflow as tf
from detailed_captioning.utils import tile_with_new_axis


class BestFirstModule(tf.layers.Layer):

    def __init__(self, word_embeddings):

        self.word_embeddings = tf.get_variable("word_embeddings", 
            initializer=tf.constant(word_embeddings, dtype=tf.float32), dtype=tf.float32)
        self.fw_cell = tf.contrib.rnn.LSTMCell(word_embeddings.shape[1])
        self.bw_cell = tf.contrib.rnn.LSTMCell(word_embeddings.shape[1])
        self.pointer_layer = tf.layers.Dense(1)
        self.logits_layer = tf.layers.Dense(word_embeddings.shape[0])


    @property
    def trainable_variables(self):

        return ([self.word_embeddings] + self.fw_cell.trainable_variables + 
            self.bw_cell.trainable_variables + self.pointer_layer.trainable_variables + 
            self.logits_layer.trainable_variables)


    @property
    def trainable_weights(self):

        return self.trainable_variables


    @property
    def variables(self):

        return ([self.word_embeddings] + self.fw_cell.variables + self.bw_cell.variables +
            self.pointer_layer.variables + self.logits_layer.variables)


    @property
    def weights(self):

        return self.variables
        

    def __call__(self, image_features, caption_ids, previous_ids, 
            indicators=None, pointer_ids=None):

        if indicators is None:
            indicators = tf.ones(tf.shape(caption_ids))
        
        # The Bidirectional RNN sequence encoder 
        caption_embeddings = tf.nn.embedding_lookup(self.word_embeddings, caption_ids)
        lengths = tf.cast(tf.reduce_sum(indicators, axis=1), tf.int32)
        outputs_tuple, _ = tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, 
            caption_embeddings, sequence_length=lengths, dtype=tf.float32)
        outputs = tf.concat(outputs_tuple, 2)
        slots = tf.concat([ outputs[:, :-1, :], outputs[:, 1:, :] ], 2)
        
        # The Pointer Network mechanism
        previous_embeddings = tf.nn.embedding_lookup(self.word_embeddings, previous_ids)
        num_slots = tf.shape(slots)[1]
        pointer_inputs = tf.concat([slots, tile_with_new_axis(
            previous_embeddings, [num_slots], [1])], 2)
        pointer_logits = tf.squeeze(self.pointer_layer(pointer_inputs))
        pointer_logits = pointer_logits * indicators[:, :-1] * indicators[:, 1:]

        if pointer_ids is None:
            pointer_ids = tf.argmax(pointer_logits, axis=1, output_type=tf.int32)

        # The word prediction mechanism
        batch_size = tf.shape(slots)[0]
        expansion_slots = tf.gather_nd(slots, tf.stack([
            tf.range(batch_size), pointer_ids], axis=1))
        word_inputs = tf.concat([ expansion_slots, image_features ], 1)
        word_logits = self.logits_layer(word_inputs)

        return pointer_logits, word_logits
