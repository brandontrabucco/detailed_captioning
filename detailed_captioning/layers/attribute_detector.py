'''Author: Brandon Trabucco, Copyright 2019
Create the general image captioner mechanism. '''


import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq


class AttributeDetector(tf.keras.layers.Layer):

    def __init__(self, num_attributes, name=None, trainable=True, **kwargs ):
        self.num_attributes = num_attributes
        self.threshold = tf.get_variable("threshold", shape=[1, num_attributes], dtype=tf.float32, 
            initializer=tf.zeros_initializer())
        super(AttributeDetector, self).__init__(name=name, trainable=trainable, **kwargs)
        self.attributes_layer = tf.layers.Dense(self.num_attributes, name="attributes_layer", 
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        
    
    def __call__(self, mean_image_features, ground_truth_attributes=None, false_positive_constraint=0.5):
        batch_size = tf.shape(mean_image_features)[0]
        logits = self.attributes_layer(mean_image_features)
        decision_boundary = tf.nn.sigmoid(self.threshold)
        attributes = tf.nn.sigmoid(logits) - decision_boundary
        attributes = ((attributes / tf.abs(attributes)) + 1.0) / 2.0
        update_hypothesis_step = None
        if ground_truth_attributes is not None:
            grad = tf.gradients(decision_boundary, self.threshold)[0]
            false_positives = tf.nn.relu(attributes - ground_truth_attributes)
            not_false_positives = 1.0 - false_positives
            adjusted_fp = tf.reduce_mean(false_positives * (1.0 - false_positive_constraint), 0)
            adjusted_not_fp = tf.reduce_mean(not_false_positives * false_positive_constraint, 0)
            ascent_step = tf.assign(self.threshold, self.threshold + grad * adjusted_fp)
            descent_step = tf.assign(self.threshold, self.threshold - grad * adjusted_not_fp)
            update_hypothesis_step = tf.group(ascent_step, descent_step)
        return logits, attributes, update_hypothesis_step
        
    @property
    def trainable_variables(self):
        return self.attributes_layer.trainable_variables + [self.threshold]
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        return self.attributes_layer.variables + [self.threshold]
    
    @property
    def weights(self):
        return self.variables
    