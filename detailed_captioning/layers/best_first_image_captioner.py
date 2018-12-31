'''Author: Brandon Trabucco, Copyright 2019
Create the general image captioner mechanism. '''


import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from detailed_captioning.cells.best_first_cell import BestFirstCell
from detailed_captioning.cells.best_first_cell import BestFirstStateTuple


class BestFirstImageCaptioner(tf.keras.layers.Layer):

    def __init__(self, image_caption_cell, word_vocabulary, word_embeddings,
            beam_size=3, maximum_iterations=100, name=None, trainable=True, **kwargs ):
        self.word_vocabulary = word_vocabulary
        self.beam_size = beam_size
        self.maximum_iterations = maximum_iterations
        self.vocab_size = word_embeddings.shape[0]
        self.embedding_size = word_embeddings.shape[1]
        super(BestFirstImageCaptioner, self).__init__(name=name, trainable=trainable, **kwargs)
        self.embeddings_map = tf.get_variable("embeddings_map", dtype=tf.float32,
            initializer=tf.constant(word_embeddings, dtype=tf.float32))
        self.best_first_cell = BestFirstCell(image_caption_cell, self.embeddings_map,
            maximum_sequence_length=maximum_iterations)
        self.logits_layer = tf.layers.Dense(self.vocab_size, name="logits_layer", 
            kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    def __call__(self, 
            mean_image_features=None, 
            mean_object_features=None, 
            spatial_image_features=None, 
            spatial_object_features=None, 
            word_ids=None, pointer_ids=None, lengths=None):
        
        assert(mean_image_features is not None or mean_object_features is not None or
            spatial_image_features is not None or spatial_object_features is not None)
        use_beam_search = (word_ids is None or pointer_ids is None or lengths is None)
        if mean_image_features is not None:
            batch_size = tf.shape(mean_image_features)[0]
        elif mean_object_features is not None:
            batch_size = tf.shape(mean_object_features)[0]
        elif spatial_image_features is not None:
            batch_size = tf.shape(spatial_image_features)[0]
        elif spatial_object_features is not None:
            batch_size = tf.shape(spatial_object_features)[0] 
        
        initial_state = BestFirstStateTuple(
            sentinel=tf.cast(tf.ones([batch_size, 1]), tf.float32),
            caption=tf.cast(tf.concat([
                    tf.fill([batch_size, 1], self.word_vocabulary.end_id), 
                    tf.zeros([batch_size, self.best_first_cell.maximum_sequence_length - 1], dtype=tf.int32)
                ], 1), tf.int32),
            running_words=tf.cast(tf.zeros([batch_size, 
                self.best_first_cell.maximum_sequence_length]), tf.int32),
            running_pointers=tf.cast(tf.zeros([batch_size, 
                self.best_first_cell.maximum_sequence_length]), tf.int32),
            lengths=tf.cast(tf.ones([batch_size, 1]), tf.int32),
            previous_pointer=tf.cast(tf.zeros([batch_size, 1]), tf.int32))
        
        self.best_first_cell.is_using_dynamic_rnn = False
        
        if use_beam_search:
            
            if mean_image_features is not None:
                mean_image_features = seq2seq.tile_batch(mean_image_features, 
                    multiplier=self.beam_size)
                self.best_first_cell.mean_image_features = mean_image_features
            if mean_object_features is not None:
                mean_object_features = seq2seq.tile_batch(mean_object_features, 
                    multiplier=self.beam_size)
                self.best_first_cell.mean_object_features = mean_object_features
            if spatial_image_features is not None:
                spatial_image_features = seq2seq.tile_batch(spatial_image_features, 
                    multiplier=self.beam_size)
                self.best_first_cell.spatial_image_features = spatial_image_features
            if spatial_object_features is not None:
                spatial_object_features = seq2seq.tile_batch(spatial_object_features, 
                    multiplier=self.beam_size)
                self.best_first_cell.spatial_object_features = spatial_object_features
                
            initial_state = seq2seq.tile_batch(initial_state, multiplier=self.beam_size)
            decoder = seq2seq.BeamSearchDecoder(self.best_first_cell, lambda x: tf.cast(x, tf.int32), 
                tf.fill([batch_size], self.word_vocabulary.start_id), self.word_vocabulary.end_id, 
                initial_state, self.beam_size, output_layer=self.logits_layer)
            _outputs, state, lengths = seq2seq.dynamic_decode(decoder, 
                maximum_iterations=self.maximum_iterations)
            
            # Unclear if the beam dimension on the inside or outside of the state
            cell_state = state.cell_state
            sequence_length = tf.shape(cell_state.running_words)[2]
            word_ids = tf.reshape(cell_state.running_words, [batch_size * self.beam_size, 
                sequence_length])
            pointer_ids = tf.reshape(cell_state.running_pointers, [batch_size * self.beam_size, 
                sequence_length])
            
        if mean_image_features is not None:
            self.best_first_cell.mean_image_features = mean_image_features
        if mean_object_features is not None:
            self.best_first_cell.mean_object_features = mean_object_features
        if spatial_image_features is not None:
            self.best_first_cell.spatial_image_features = spatial_image_features
        if spatial_object_features is not None:
            self.best_first_cell.spatial_object_features = spatial_object_features   
            
        self.best_first_cell.is_using_dynamic_rnn = True
        
        if use_beam_search:
            rnn_inputs = tf.concat([tf.expand_dims(word_ids[:, :-1], 2), 
                tf.expand_dims(pointer_ids[:, 1:], 2)], 2)
        else:
            rnn_inputs = tf.concat([tf.expand_dims(word_ids, 2), tf.expand_dims(pointer_ids, 2)], 2)
        activations, _state = tf.nn.dynamic_rnn(self.best_first_cell, rnn_inputs, 
            sequence_length=tf.reshape(lengths, [-1]), initial_state=initial_state, dtype=tf.float32)
        word_activations, pointer_logits = tf.split(activations, [
            self.best_first_cell.output_size - self.maximum_iterations, 
            self.maximum_iterations], axis=2)
        word_logits = self.logits_layer(word_activations)
        if use_beam_search:
            length = tf.shape(word_logits)[1]
            word_logits = tf.reshape(word_logits, [batch_size, self.beam_size, length, self.vocab_size])
            pointer_logits = tf.reshape(pointer_logits, [batch_size, self.beam_size, length, 
                self.maximum_iterations])
            
        return (word_logits, tf.argmax(word_logits, axis=-1, output_type=tf.int32),
                pointer_logits, tf.argmax(pointer_logits, axis=-1, output_type=tf.int32))
        
    @property
    def trainable_variables(self):
        return (self.best_first_cell.trainable_variables 
                + self.logits_layer.trainable_variables 
                + [self.embeddings_map])
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        return (self.best_first_cell.variables 
                + self.logits_layer.variables 
                + [self.embeddings_map])
    
    @property
    def weights(self):
        return self.variables
    