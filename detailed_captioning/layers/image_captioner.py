'''Author: Brandon Trabucco, Copyright 2019
Create the bottom-up top-down image captioner using the 
TF Object Detection API.'''


import collections
import tensorflow as tf
from detailed_captioning.layers.feature_extractor import FeatureExtractor
from detailed_captioning.layers.up_down_cell import UpDownCell
from detailed_captioning.utils import load_glove


# Used to store the variables for the detector and captioner
_CaptionerVariables = collections.namedtuple("CaptionerVariables", ("feature_extractor", "up_down_cell"))
class CaptionerVariables(_CaptionerVariables):
    __slots__ = ()
    @property
    def all(self):
        return self.feature_extractor + self.up_down_cell


def _repeat_elements(num_elements, num_repeats):
    # Tensor: [[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, ...]]
    return tf.reshape(tf.tile(tf.expand_dims(tf.range(num_elements), 1), 
                              [1, num_repeats]), [-1])


class ImageCaptioner(tf.keras.layers.Layer):
    '''Create the bottom-up top-down image captioner using the 
    TF Object Detection API.'''

    def __init__(self, lstm_units, 
            batch_size=1, beam_size=3, vocab_size=1000, embedding_size=50, 
            name=None, trainable=True, fine_tune_cnn=False, 
            use_peepholes=False, cell_clip=None,
            initializer=None, num_proj=None, proj_clip=None,
            num_unit_shards=None, num_proj_shards=None,
            forget_bias=1.0, state_is_tuple=True,
            activation=None, reuse=None, dtype=None,
            attention_method='softmax', **kwargs ):
    
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        super(ImageCaptioner, self).__init__(name=name, trainable=trainable, 
            **kwargs)
        # Load the ResNet-101 CNN Model using Keras Layer API
        self.feature_extractor = FeatureExtractor(is_training=fine_tune_cnn, reuse=reuse)
        # Load the Up Down RNN Cell to decode with.
        self.up_down_cell = UpDownCell(lstm_units, 
            use_peepholes=use_peepholes, cell_clip=cell_clip,
            initializer=initializer, num_proj=num_proj, proj_clip=proj_clip,
            num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
            forget_bias=forget_bias, state_is_tuple=state_is_tuple,
            activation=activation, reuse=reuse, name=name, dtype=dtype,
            # The extra parameters for the up-down mechanism
            image_features=None, object_features=None,
            attention_method=attention_method, **kwargs )
        # Load the pretrained GloVE
        vocab, pretrained_matrix = load_glove(vocab_size, embedding_size)
        self.vocab = vocab
        self.embeddings_map = tf.get_variable("embeddings_map", dtype=tf.float32,
            initializer=tf.constant(pretrained_matrix, dtype=tf.float32), trainable=False)
        self.logits_layer = tf.layers.Dense(vocab_size, name="logits_layer", 
            kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    def __call__(self, images, boxes, cropped_images=None, seq_inputs=None, lengths=None):
        
        # Crop inputs according to the ROI bounding boxes
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        # Maybe need to compute the cropped images (eg: during training)
        num_boxes = tf.shape(boxes)[1]
        if cropped_images is None:
            cropped_images = tf.image.crop_and_resize(images, tf.reshape(boxes, [-1, 4]), 
                _repeat_elements(batch_size, num_boxes), [height, width])
        # Run the object detector to extract image and ROI features
        image_features = tf.reduce_mean(self.feature_extractor(images), [1, 2])
        object_features = tf.reduce_mean(self.feature_extractor(cropped_images), [1, 2])
        # There are 2048 dims in the channel since we are using ResNet-101
        object_features = tf.reshape(object_features, [batch_size, num_boxes, 2048])
        initial_state = self.up_down_cell.zero_state(self.batch_size, tf.float32)
        # Perform beam search if not provided the ground truth
        if seq_inputs is None or lengths is None:
            image_features = tf.contrib.seq2seq.tile_batch(
                image_features, multiplier=self.beam_size)
            object_features = tf.contrib.seq2seq.tile_batch(
                object_features, multiplier=self.beam_size)
            self.up_down_cell.image_features = image_features
            self.up_down_cell.object_features = object_features
            # Load the beam search decoder for inference
            initial_state = tf.contrib.seq2seq.tile_batch(
                initial_state, multiplier=self.beam_size)
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                self.up_down_cell, self.embeddings_map, 
                tf.fill([self.batch_size], self.vocab.start_id), self.vocab.end_id, 
                initial_state, self.beam_size, output_layer=self.logits_layer)
            # Run the Up-Down RNN Cell to obtain the beam search captions
            outputs, state, lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder, maximum_iterations=100)
            ids = tf.transpose(outputs.predicted_ids, [0, 2, 1])
            sequence_length = tf.shape(ids)[2]
            flat_ids = tf.reshape(ids, [self.batch_size * self.beam_size, sequence_length])
            seq_inputs = tf.concat([
                tf.fill([self.batch_size * self.beam_size, 1], self.vocab.start_id), 
                flat_ids], 1)
        # Workaround to obtain the actual logits for beam search
        self.up_down_cell.image_features = image_features
        self.up_down_cell.object_features = object_features
        flat_logits, _ = tf.nn.dynamic_rnn(
            self.up_down_cell, tf.nn.embedding_lookup(self.embeddings_map, seq_inputs),
            sequence_length=tf.reshape(lengths, [-1]), initial_state=initial_state)
        logits = self.logits_layer(flat_logits)
        if seq_inputs is None or lengths is None:
            logits = tf.reshape(logits, [self.batch_size, self.beam_size, 
                                         sequence_length, self.vocab_size])
        ids = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return logits, ids
        
    @property
    def trainable_variables(self):
        # Returns first the object detector variables and second the captioner variables
        all_variables = (self.logits_layer.trainable_variables + [self.embeddings_map])
        return CaptionerVariables(self.feature_extractor.trainable_variables, 
            self.up_down_cell.trainable_variables + all_variables)
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        # Returns first the object detector variables and second the captioner variables
        all_variables = (self.logits_layer.variables + [self.embeddings_map])
        return CaptionerVariables(self.feature_extractor.variables, 
            self.up_down_cell.variables + all_variables)
    
    @property
    def weights(self):
        return self.variables
    