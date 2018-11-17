'''Author: Brandon Trabucco, Copyright 2019
An abstract class for implementing an RNN cell that generates 
sentence captions from image features.'''


import tensorflow as tf
from tensorflow.python.keras import initializers
from detailed_captioning.utils import tile_with_new_axis


class ImageCaptionCell(tf.contrib.rnn.LayerRNNCell):

    def __init__(self, 
            reuse=None, name=None, dtype=None,
            mean_image_features=None, mean_object_features=None, 
            spatial_image_features=None, spatial_object_features=None, 
            **kwargs ):
        super(ImageCaptionCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        self._mean_image_features = mean_image_features
        self._mean_object_features = mean_object_features
        self._spatial_image_features = spatial_image_features
        self._spatial_object_features = spatial_object_features
        
    @property
    def mean_image_features(self):
        return self._mean_image_features
    
    @property
    def mean_object_features(self):
        return self._mean_object_features
    
    @mean_image_features.setter
    def mean_image_features(self, x):
        self._mean_image_features = x
    
    @mean_object_features.setter
    def mean_object_features(self, x):
        self._mean_object_features = x
        
    @property
    def spatial_image_features(self):
        return self._spatial_image_features
    
    @property
    def spatial_object_features(self):
        return self._spatial_object_features
    
    @spatial_image_features.setter
    def spatial_image_features(self, x):
        self._spatial_image_features = x
    
    @spatial_object_features.setter
    def spatial_object_features(self, x):
        self._spatial_object_features = x

    @property
    def state_size(self):
        raise NotImplementedError()

    @property
    def output_size(self):
        raise NotImplementedError()

    def __call__(self, inputs, state):
        raise NotImplementedError()
    
    @property
    def trainable_variables(self):
        raise NotImplementedError()
    
    @property
    def trainable_weights(self):
        raise NotImplementedError()
    
    @property
    def variables(self):
        raise NotImplementedError()
    
    @property
    def weights(self):
        raise NotImplementedError()
    
