'''Author: Brandon Trabucco, Copyright 2019
Create the general image captioner mechanism. '''


import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq


class AttributeDetector(tf.keras.layers.Layer):

    def __init__(self, num_attributes, threshold=0.9, name=None, trainable=True, **kwargs ):
        self.num_attributes = num_attributes
        self.threshold = threshold
        super(AttributeDetector, self).__init__(name=name, trainable=trainable, **kwargs)
        self.attributes_layer = tf.layers.Dense(self.num_attributes, name="attributes_layer", 
            kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    def __call__(self, mean_image_features):
        batch_size = tf.shape(mean_image_features)[0]
        logits = self.attributes_layer(mean_image_features)
        attributes = tf.nn.sigmoid(logits) - self.threshold
        attributes = attributes / tf.abs(attributes)
        attributes = (attributes + 1.0) / 2.0
        return logits, attributes
        
    @property
    def trainable_variables(self):
        return self.attributes_layer.trainable_variables
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        return self.attributes_layer.variables
    
    @property
    def weights(self):
        return self.variables
    