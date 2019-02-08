'''Author: Brandon Trabucco, Copyright 2019
An abstract class for implementing an RNN cell that generates 
sentence captions from image features and attributes.'''


import tensorflow as tf
from detailed_captioning.cells.image_caption_cell import ImageCaptionCell


class AttributeCaptionCell(ImageCaptionCell):

    def __init__(self, 
            reuse=None, name=None, dtype=None,
            mean_image_features=None, mean_object_features=None, 
            spatial_image_features=None, spatial_object_features=None, 
            image_attribute_features=None, object_attribute_features=None, 
            num_image_features=2048, num_attribute_features=300,
            **kwargs ):
        super(AttributeCaptionCell, self).__init__(
            reuse=reuse, name=name, dtype=dtype,
            mean_image_features=mean_image_features, 
            mean_object_features=mean_object_features, 
            spatial_image_features=spatial_image_features, 
            spatial_object_features=spatial_object_features, 
            num_image_features=num_image_features, **kwargs )
        self._image_attribute_features = image_attribute_features
        self._object_attribute_features = object_attribute_features
        self.num_attribute_features = num_attribute_features
        
    @property
    def image_attribute_features(self):
        return self._image_attribute_features
    
    @property
    def object_attribute_features(self):
        return self._object_attribute_features
    
    @image_attribute_features.setter
    def image_attribute_features(self, x):
        self._image_attribute_features = x
    
    @object_attribute_features.setter
    def object_attribute_features(self, x):
        self._object_attribute_features = x
    
