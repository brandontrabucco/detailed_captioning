'''Author: Brandon Trabucco, Copyright 2019
An attribute detector consisting of a single dense layer. '''


import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq


class AttributeDetector(tf.keras.layers.Layer):

    def __init__(self, num_attributes, name=None, trainable=True, **kwargs ):
        self.num_attributes = num_attributes
        super(AttributeDetector, self).__init__(name=name, trainable=trainable, **kwargs)
        self.attributes_layer = tf.layers.Dense(self.num_attributes, name="attributes_layer", 
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        
    
    def __call__(self, mean_image_features, mean_object_features, k=8):
        image_logits = self.attributes_layer(mean_image_features)
        object_logits = self.attributes_layer(mean_object_features)
        logits = tf.reduce_max(tf.concat([tf.expand_dims(image_logits, 1), object_logits], 1), 1)
        _, image_attributes = tf.nn.top_k(logits, k=k)
        _, object_attributes = tf.nn.top_k(object_logits, k=k)
        return logits, image_attributes, object_attributes
        
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
    