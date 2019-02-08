'''Author: Brandon Trabucco, Copyright 2019
Create a mechanism that bridges attributes with image and object features '''


import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from detailed_captioning.utils import collapse_dims


class AttributeCaptioner(tf.keras.layers.Layer):

    def __init__(self, attribute_caption_cell, word_vocabulary, word_embeddings,
            attribute_to_word_lookup_table,
            beam_size=3, maximum_iterations=100,
            name=None, trainable=True, **kwargs ):
        self.attribute_caption_cell = attribute_caption_cell
        self.word_vocabulary = word_vocabulary
        self.vocab_size = word_embeddings.shape[0]
        self.attribute_to_word_lookup_table = tf.constant(attribute_to_word_lookup_table)
        self.beam_size = beam_size
        self.maximum_iterations = maximum_iterations
        super(AttributeCaptioner, self).__init__(name=name, trainable=trainable, **kwargs)
        self.word_embeddings_map = tf.get_variable("word_embeddings_map", dtype=tf.float32,
            initializer=tf.constant(word_embeddings, dtype=tf.float32))
        self.word_logits_layer = tf.layers.Dense(self.vocab_size, name="word_logits_layer", 
            kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    def __call__(self, 
            image_attributes=None,
            object_attributes=None,
            mean_image_features=None, 
            mean_object_features=None, 
            spatial_image_features=None, 
            spatial_object_features=None, 
            seq_inputs=None, lengths=None ):
        assert(image_attributes is not None or object_attributes is not None or
            mean_image_features is not None or mean_object_features is not None or
            spatial_image_features is not None or spatial_object_features is not None)
        use_beam_search = (seq_inputs is None or lengths is None)
        if image_attributes is not None:
            batch_size = tf.shape(image_attributes)[0]
            image_attribute_ids = tf.gather(self.attribute_to_word_lookup_table, image_attributes)
            image_attribute_features = tf.nn.embedding_lookup(self.word_embeddings_map, 
                image_attribute_ids)
        if object_attributes is not None:
            batch_size = tf.shape(object_attributes)[0]
            object_attribute_ids = tf.gather(self.attribute_to_word_lookup_table, object_attributes)
            object_attribute_features = tf.nn.embedding_lookup(self.word_embeddings_map, 
                object_attribute_ids)
        if mean_image_features is not None:
            batch_size = tf.shape(mean_image_features)[0]
        if mean_object_features is not None:
            batch_size = tf.shape(mean_object_features)[0]
        if spatial_image_features is not None:
            batch_size = tf.shape(spatial_image_features)[0]
            spatial_image_features = collapse_dims(spatial_image_features, [1, 2])
            mean_image_features = tf.concat([tf.reduce_mean(spatial_image_features, [1]), 
                mean_attribute_features], 1)
        if spatial_object_features is not None:
            batch_size = tf.shape(spatial_object_features)[0] 
            spatial_object_features = collapse_dims(spatial_object_features, [2, 3])
            mean_object_features = tf.concat([tf.reduce_mean(spatial_object_features, [2]), 
                attribute_features], 1)
        initial_state = self.attribute_caption_cell.zero_state(batch_size, tf.float32)
        if use_beam_search:
            if image_attribute_features is not None:
                image_attribute_features = seq2seq.tile_batch(image_attribute_features, 
                    multiplier=self.beam_size)
                self.attribute_caption_cell.image_attribute_features = image_attribute_features
            if object_attribute_features is not None:
                object_attribute_features = seq2seq.tile_batch(object_attribute_features, 
                    multiplier=self.beam_size)
                self.attribute_caption_cell.object_attribute_features = object_attribute_features
            if mean_image_features is not None:
                mean_image_features = seq2seq.tile_batch(mean_image_features, 
                    multiplier=self.beam_size)
                self.attribute_caption_cell.mean_image_features = mean_image_features
            if mean_object_features is not None:
                mean_object_features = seq2seq.tile_batch(mean_object_features, 
                    multiplier=self.beam_size)
                self.attribute_caption_cell.mean_object_features = mean_object_features
            if spatial_image_features is not None:
                spatial_image_features = seq2seq.tile_batch(spatial_image_features, 
                    multiplier=self.beam_size)
                self.attribute_caption_cell.spatial_image_features = spatial_image_features
            if spatial_object_features is not None:
                spatial_object_features = seq2seq.tile_batch(spatial_object_features, 
                    multiplier=self.beam_size)
                self.attribute_caption_cell.spatial_object_features = spatial_object_features
            initial_state = seq2seq.tile_batch(initial_state, multiplier=self.beam_size)
            decoder = seq2seq.BeamSearchDecoder(self.attribute_caption_cell, self.word_embeddings_map, 
                tf.fill([batch_size], self.word_vocabulary.start_id), self.word_vocabulary.end_id, 
                initial_state, self.beam_size, output_layer=self.word_logits_layer)
            outputs, state, lengths = seq2seq.dynamic_decode(decoder, 
                maximum_iterations=self.maximum_iterations)
            ids = tf.transpose(outputs.predicted_ids, [0, 2, 1])
            sequence_length = tf.shape(ids)[2]
            flat_ids = tf.reshape(ids, [batch_size * self.beam_size, sequence_length])
            seq_inputs = tf.concat([
                tf.fill([batch_size * self.beam_size, 1], self.word_vocabulary.start_id), flat_ids], 1)
        if image_attribute_features is not None:
            self.attribute_caption_cell.image_attribute_features = image_attribute_features
        if object_attribute_features is not None:
            self.attribute_caption_cell.object_attribute_features = object_attribute_features
        if mean_image_features is not None:
            self.attribute_caption_cell.mean_image_features = mean_image_features
        if mean_object_features is not None:
            self.attribute_caption_cell.mean_object_features = mean_object_features
        if spatial_image_features is not None:
            self.attribute_caption_cell.spatial_image_features = spatial_image_features
        if spatial_object_features is not None:
            self.attribute_caption_cell.spatial_object_features = spatial_object_features   
        activations, _state = tf.nn.dynamic_rnn(self.attribute_caption_cell, 
            tf.nn.embedding_lookup(self.word_embeddings_map, seq_inputs),
            sequence_length=tf.reshape(lengths, [-1]), initial_state=initial_state)
        logits = self.word_logits_layer(activations)
        if use_beam_search:
            length = tf.shape(logits)[1]
            logits = tf.reshape(logits, [batch_size, self.beam_size, length, self.vocab_size])
        return logits, tf.argmax(logits, axis=-1, output_type=tf.int32)
        
    @property
    def trainable_variables(self):
        return (self.attribute_caption_cell.trainable_variables 
                + self.word_logits_layer.trainable_variables 
                + [self.word_embeddings_map])
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        return (self.attribute_caption_cell.variables 
                + self.word_logits_layer.variables 
                + [self.word_embeddings_map])
    
    @property
    def weights(self):
        return self.variables
    