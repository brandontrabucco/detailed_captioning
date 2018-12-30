"""Author: Brandon Trabucco, Copyright 2019
Implements the Best First Module for image captioning."""


import tensorflow as tf
from detailed_captioning.utils import tile_with_new_axis


class BestFirstModule(tf.layers.Layer):

    def __init__(self, image_caption_cell, word_vocabulary, word_embeddings):

        self.word_vocabulary = word_vocabulary
        self.embeddings_map = tf.get_variable("embeddings_map", 
            initializer=tf.constant(word_embeddings, dtype=tf.float32), dtype=tf.float32)
        self.image_caption_cell = image_caption_cell
        self.pointer_layer = tf.layers.Dense(1)
        self.logits_layer = tf.layers.Dense(word_embeddings.shape[0])


    @property
    def trainable_variables(self):

        return ([self.word_embeddings] + self.image_caption_cell.trainable_variables + 
            self.pointer_layer.trainable_variables + self.logits_layer.trainable_variables)


    @property
    def trainable_weights(self):

        return self.trainable_variables


    @property
    def variables(self):

        return ([self.word_embeddings] + self.image_caption_cell.variables +
            self.pointer_layer.variables + self.logits_layer.variables)


    @property
    def weights(self):

        return self.variables
        

    def __call__(self,
            caption_ids, indicators=None,
            mean_image_features=None, 
            mean_object_features=None, 
            spatial_image_features=None, 
            spatial_object_features=None,
            pointer_ids=None):
        
        assert(mean_image_features is not None or mean_object_features is not None or
            spatial_image_features is not None or spatial_object_features is not None)
        
        if mean_image_features is not None:
            batch_size = tf.shape(mean_image_features)[0]
        elif mean_object_features is not None:
            batch_size = tf.shape(mean_object_features)[0]
        elif spatial_image_features is not None:
            batch_size = tf.shape(spatial_image_features)[0]
        elif spatial_object_features is not None:
            batch_size = tf.shape(spatial_object_features)[0] 
        
        if mean_image_features is not None:
            self.image_caption_cell.mean_image_features = mean_image_features
        if mean_object_features is not None:
            self.image_caption_cell.mean_object_features = mean_object_features
        if spatial_image_features is not None:
            self.image_caption_cell.spatial_image_features = spatial_image_features
        if spatial_object_features is not None:
            self.image_caption_cell.spatial_object_features = spatial_object_features

        if indicators is None:
            indicators = tf.ones(tf.shape(caption_ids))
            
        # The Image-Caption RNN sequence encoder 
        caption_embeddings = tf.nn.embedding_lookup(self.word_embeddings, caption_ids)
        lengths = tf.cast(tf.reduce_sum(indicators, axis=1), tf.int32)
        slots, _states = tf.nn.dynamic_rnn(self.image_caption_cell, caption_embeddings, 
            sequence_length=lengths, dtype=tf.float32)
        
        # The Pointer mechanism
        num_slots = tf.shape(slots)[1]
        index_features = tile_with_new_axis(tf.range(tf.to_float(num_slots)), [batch_size], [
            0]) / tf.to_float(tf.expand_dims(lengths, 1)) * indicators
        pointer_inputs = tf.concat([slots, tf.expand_dims(index_features, 2)], 2)
        pointer_logits = tf.squeeze(self.pointer_layer(pointer_inputs))

        if pointer_ids is None:
            pointer_ids = tf.argmax(pointer_logits, axis=1, output_type=tf.int32)

        # The word prediction mechanism
        expansion_slots = tf.gather_nd(slots, tf.stack([tf.range(batch_size), pointer_ids], axis=1))
        word_logits = self.logits_layer(expansion_slots)
        word_logits = (word_logits + tf.stop_gradient(tf.reduce_min(word_logits))) * indicators

        return pointer_logits, word_logits
