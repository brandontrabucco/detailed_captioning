'''Author: Brandon Trabucco, Copyright 2019
Creates the ResNet-101 CNN architecture to process image features.
https://arxiv.org/abs/1603.05027'''


import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_101
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_arg_scope


class FeatureExtractor(tf.keras.layers.Layer):
    '''Creates the ResNet-101 CNN architecture to process image features.
    https://arxiv.org/abs/1603.05027'''

    def __init__(self, num_classes=None, is_training=True, 
            global_pool=True, output_stride=None, reuse=False, 
            scope='resnet_v2_101', **kwargs):
    
        self.num_classes = num_classes
        self.is_training = is_training
        self.global_pool = global_pool
        self.output_stride = output_stride
        self.reuse = reuse
        self.scope = scope
        super(FeatureExtractor, self).__init__(name=self.scope, trainable=is_training, **kwargs)
    
    def __call__(self, inputs):
        
        inputs = ((inputs / 255.0) - 0.5) * 2.0
        
        with tf.contrib.slim.arg_scope(resnet_arg_scope()):
            
            image_features, end_points = resnet_v2_101(inputs, 
                num_classes=self.num_classes, is_training=self.is_training,
                global_pool=self.global_pool, output_stride=self.output_stride,
                reuse=self.reuse, scope=self.scope)
            self.reuse = True
                
        return image_features
        
    @property
    def trainable_variables(self):
        detector_variables = [x for x in tf.trainable_variables() if self.scope + '/' in x.name]
        return detector_variables
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        detector_variables = [x for x in tf.global_variables() if self.scope + '/' in x.name]
        return detector_variables
    
    @property
    def weights(self):
        return self.variables