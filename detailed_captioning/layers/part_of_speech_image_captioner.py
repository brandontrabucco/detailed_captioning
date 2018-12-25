'''Author: Brandon Trabucco, Copyright 2019
Create the general image captioner mechanism that first decodes part of speech. '''


import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq


class PartOfSpeechImageCaptioner(tf.keras.layers.Layer):

    def __init__(self, image_caption_cell, word_vocabulary, word_embeddings,
            pos_decoder_cell, pos_encoder_cell, pos_vocabulary, pos_embeddings,
            beam_size=3, maximum_iterations=100,
            name=None, trainable=True, **kwargs ):
        # The Image Caption Cell objects
        self.image_caption_cell = image_caption_cell
        self.pos_decoder_cell = pos_decoder_cell
        self.pos_encoder_cell = pos_encoder_cell
        # Other special parameters
        self.word_vocabulary = word_vocabulary
        self.pos_vocabulary = pos_vocabulary
        self.vocab_size = word_embeddings.shape[0]
        self.pos_size = pos_embeddings.shape[0]
        self.beam_size = beam_size
        self.maximum_iterations = maximum_iterations
        # Setup the TF layer object
        super(PartOfSpeechImageCaptioner, self).__init__(name=name, trainable=trainable, **kwargs)
        self.word_embeddings_map = tf.get_variable("word_embeddings_map", dtype=tf.float32,
            initializer=tf.constant(word_embeddings, dtype=tf.float32))
        self.pos_embeddings_map = tf.get_variable("pos_embeddings_map", dtype=tf.float32,
            initializer=tf.constant(pos_embeddings, dtype=tf.float32))
        self.word_logits_layer = tf.layers.Dense(self.vocab_size, name="word_logits_layer", 
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.pos_logits_layer = tf.layers.Dense(self.pos_size, name="pos_logits_layer", 
            kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    def __call__(self, 
            mean_image_features=None, 
            mean_object_features=None, 
            spatial_image_features=None, 
            spatial_object_features=None, 
            word_seq_inputs=None, word_lengths=None, 
            pos_seq_inputs=None, pos_seq_outputs=None, pos_lengths=None):
        assert(mean_image_features is not None or mean_object_features is not None or
            spatial_image_features is not None or spatial_object_features is not None)
        use_beam_search = (word_seq_inputs is None or word_lengths is None or 
            pos_seq_inputs is None or pos_seq_outputs is None or pos_lengths is None)
        if mean_image_features is not None:
            batch_size = tf.shape(mean_image_features)[0]
        elif mean_object_features is not None:
            batch_size = tf.shape(mean_object_features)[0]
        elif spatial_image_features is not None:
            batch_size = tf.shape(spatial_image_features)[0]
        elif spatial_object_features is not None:
            batch_size = tf.shape(spatial_object_features)[0] 
        pos_decoder_initial_state = self.pos_decoder_cell.zero_state(batch_size, tf.float32)
        pos_encoder_initial_state = self.pos_encoder_cell.zero_state(batch_size, tf.float32)
        if use_beam_search:
            if mean_image_features is not None:
                mean_image_features = seq2seq.tile_batch(mean_image_features, 
                    multiplier=self.beam_size)
                self.image_caption_cell.mean_image_features = mean_image_features
                self.pos_decoder_cell.mean_image_features = mean_image_features
                self.pos_encoder_cell.mean_image_features = mean_image_features
            if mean_object_features is not None:
                mean_object_features = seq2seq.tile_batch(mean_object_features, 
                    multiplier=self.beam_size)
                self.image_caption_cell.mean_object_features = mean_object_features
                self.pos_decoder_cell.mean_object_features = mean_object_features
                self.pos_encoder_cell.mean_object_features = mean_object_features
            if spatial_image_features is not None:
                spatial_image_features = seq2seq.tile_batch(spatial_image_features, 
                    multiplier=self.beam_size)
                self.image_caption_cell.spatial_image_features = spatial_image_features
                self.pos_decoder_cell.spatial_image_features = spatial_image_features
                self.pos_encoder_cell.spatial_image_features = spatial_image_features
            if spatial_object_features is not None:
                spatial_object_features = seq2seq.tile_batch(spatial_object_features, 
                    multiplier=self.beam_size)
                self.image_caption_cell.spatial_object_features = spatial_object_features  
                self.pos_decoder_cell.spatial_object_features = spatial_object_features
                self.pos_encoder_cell.spatial_object_features = spatial_object_features
            pos_decoder_initial_state = seq2seq.tile_batch(pos_decoder_initial_state, 
                multiplier=self.beam_size)
            pos_encoder_initial_state = seq2seq.tile_batch(pos_encoder_initial_state, 
                multiplier=self.beam_size)
            # Decode the part of speech sentence template
            pos_decoder = seq2seq.BeamSearchDecoder(self.pos_decoder_cell, self.pos_embeddings_map, 
                tf.fill([batch_size], self.pos_vocabulary.start_id), self.pos_vocabulary.end_id, 
                pos_decoder_initial_state, self.beam_size, output_layer=self.pos_logits_layer)
            pos_outputs, _state, pos_lengths = seq2seq.dynamic_decode(pos_decoder, 
                maximum_iterations=self.maximum_iterations)
            pos_ids = tf.transpose(pos_outputs.predicted_ids, [0, 2, 1])
            pos_sequence_length = tf.shape(pos_ids)[2]
            pos_seq_outputs = tf.reshape(pos_ids, [batch_size * self.beam_size, pos_sequence_length])
            pos_seq_inputs = tf.concat([
                tf.fill([batch_size * self.beam_size, 1], self.pos_vocabulary.start_id), 
                pos_seq_outputs], 1)
            # Encode the template and generate a sentence using beam search
            _activations, word_initial_state = tf.nn.dynamic_rnn(self.pos_encoder_cell, 
                tf.nn.embedding_lookup(self.pos_embeddings_map, pos_seq_outputs),
                sequence_length=tf.reshape(pos_lengths, [-1]), initial_state=pos_encoder_initial_state)
            word_decoder = seq2seq.BeamSearchDecoder(self.image_caption_cell, self.word_embeddings_map, 
                tf.fill([batch_size], self.word_vocabulary.start_id), self.word_vocabulary.end_id, 
                word_initial_state, self.beam_size, output_layer=self.word_logits_layer)
            word_outputs, _state, word_lengths = seq2seq.dynamic_decode(word_decoder, 
                maximum_iterations=self.maximum_iterations)
            word_ids = tf.transpose(word_outputs.predicted_ids, [0, 2, 1])
            word_sequence_length = tf.shape(word_ids)[2]
            word_seq_outputs = tf.reshape(word_ids, [batch_size * self.beam_size, word_sequence_length])
            word_seq_inputs = tf.concat([
                tf.fill([batch_size * self.beam_size, 1], self.word_vocabulary.start_id), 
                word_seq_outputs], 1)
        if mean_image_features is not None:
            self.image_caption_cell.mean_image_features = mean_image_features
            self.pos_decoder_cell.mean_image_features = mean_image_features
            self.pos_encoder_cell.mean_image_features = mean_image_features
        if mean_object_features is not None:
            self.image_caption_cell.mean_object_features = mean_object_features
            self.pos_decoder_cell.mean_object_features = mean_object_features
            self.pos_encoder_cell.mean_object_features = mean_object_features
        if spatial_image_features is not None:
            self.image_caption_cell.spatial_image_features = spatial_image_features
            self.pos_decoder_cell.spatial_image_features = spatial_image_features
            self.pos_encoder_cell.spatial_image_features = spatial_image_features
        if spatial_object_features is not None:
            self.image_caption_cell.spatial_object_features = spatial_object_features   
            self.pos_decoder_cell.spatial_object_features = spatial_object_features
            self.pos_encoder_cell.spatial_object_features = spatial_object_features
        pos_decoder_activations, _state = tf.nn.dynamic_rnn(self.pos_decoder_cell, 
            tf.nn.embedding_lookup(self.pos_embeddings_map, pos_seq_inputs),
            sequence_length=tf.reshape(pos_lengths, [-1]), initial_state=pos_decoder_initial_state)
        pos_logits = self.pos_logits_layer(pos_decoder_activations)
        if not use_beam_search:
            _activations, word_initial_state = tf.nn.dynamic_rnn(self.pos_encoder_cell, 
                tf.nn.embedding_lookup(self.pos_embeddings_map, pos_seq_outputs),
                sequence_length=tf.reshape(pos_lengths, [-1]), initial_state=pos_encoder_initial_state)
        word_activations, _state = tf.nn.dynamic_rnn(self.image_caption_cell, 
            tf.nn.embedding_lookup(self.word_embeddings_map, word_seq_inputs),
            sequence_length=tf.reshape(word_lengths, [-1]), initial_state=word_initial_state)
        word_logits = self.word_logits_layer(word_activations)
        if use_beam_search:
            pos_logits = tf.reshape(pos_logits, [batch_size, self.beam_size, 
                tf.shape(pos_logits)[1], self.pos_size])
            word_logits = tf.reshape(word_logits, [batch_size, self.beam_size, 
                tf.shape(word_logits)[1], self.vocab_size])
        pos_logits_ids = tf.argmax(pos_logits, axis=-1, output_type=tf.int32)
        word_logits_ids = tf.argmax(word_logits, axis=-1, output_type=tf.int32)
        return pos_logits, pos_logits_ids, word_logits, word_logits_ids
        
    @property
    def trainable_variables(self):
        return (self.image_caption_cell.trainable_variables 
                + self.pos_decoder_cell.trainable_variables 
                + self.pos_encoder_cell.trainable_variables 
                + self.word_logits_layer.trainable_variables 
                + self.pos_logits_layer.trainable_variables 
                + [self.word_embeddings_map] + [self.pos_embeddings_map])
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        return (self.image_caption_cell.variables 
                + self.pos_decoder_cell.variables 
                + self.pos_encoder_cell.variables 
                + self.word_logits_layer.variables 
                + self.pos_logits_layer.variables 
                + [self.word_embeddings_map] + [self.pos_embeddings_map])
    
    @property
    def weights(self):
        return self.variables
    