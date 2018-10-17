'''Author: Brandon Trabucco, Copyright 2019
Keras Layer API wrapper for training a TF Object Detector model.
Huang, Jonathan. et al. https://arxiv.org/abs/1611.1001'''


import tensorflow as tf
from object_detection.utils.config_util import get_configs_from_pipeline_file
from object_detection.builders import model_builder


def _repeat_elements(num_elements, num_repeats):
    # Tensor: [[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, ...]]
    return tf.reshape(tf.tile(tf.expand_dims(tf.range(num_elements), 1), 
                              [1, num_repeats]), [-1])


def _remap_name(name, namespace):
    # Remove the detector namespace and the operator index :0
    return name.replace(namespace + '/', '')[0:-2]


class ObjectDetector(tf.keras.layers.Layer):
    '''Implments a mechanism for extracting bounding boxes of objects, 
    and the image/object features thereof.
    Huang, Jonathan. et al. https://arxiv.org/abs/1611.1001'''

    def __init__(self, pipeline_config_file, name=None, trainable=True, **kwargs):
    
        super(ObjectDetector, self).__init__(name=name, trainable=trainable, 
                                             **kwargs)
        self.namespace = 'detector' + ('_' + name if name is not None else '')
        pipeline_config = get_configs_from_pipeline_file(pipeline_config_file)
        with tf.variable_scope(self.namespace, reuse=tf.AUTO_REUSE):
            self.detection_model = model_builder.build(
                pipeline_config['model'], is_training=trainable)
    
    def __call__(self, inputs):
        
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        with tf.variable_scope(self.namespace, reuse=tf.AUTO_REUSE):
            # Run the original image though the CNN and ROI detector
            preprocessed_inputs, true_shapes = self.detection_model.preprocess(
                inputs)
            features_dict = self.detection_model.predict(preprocessed_inputs, 
                                                         true_shapes)
            detector_dict = self.detection_model.postprocess(features_dict, 
                                                             true_shapes)
            # Collect the feature and ROI outputs
            image_features = features_dict['feature_maps'][-1]
            mean_image_features = tf.reduce_mean(image_features, [1, 2])
            boxes = detector_dict['detection_boxes']
            num_boxes = tf.shape(boxes)[1]
            # Crop inputs according to the ROI bounding boxes
            cropped_inputs = tf.image.crop_and_resize(
                inputs, tf.reshape(boxes, [-1, 4]), _repeat_elements(
                    batch_size, num_boxes), [height, width])
        with tf.variable_scope(self.namespace, reuse=tf.AUTO_REUSE):
            # Run the cropped ROI images through the CNN
            preprocessed_inputs, true_shapes = self.detection_model.preprocess(
                cropped_inputs)
            features_dict = self.detection_model.predict(preprocessed_inputs, 
                                                         true_shapes)
            # Collect the feature outputs
            object_features = features_dict['feature_maps'][-1]
            mean_object_features = tf.reduce_mean(object_features, [1, 2])
            mean_object_features = tf.reshape(mean_object_features, [
                batch_size, num_boxes, 256])
            return mean_image_features, boxes, mean_object_features
        
    @property
    def trainable_variables(self):
        # Handles to the models internal shared trainable variables
        detector_variables = [x for x in tf.trainable_variables() if (
            self.namespace + '/' in x.name)]
        return {_remap_name(x.name, self.namespace) : x for x in detector_variables}
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        # Handles to the models internal shared trainable variables
        detector_variables = [x for x in tf.global_variables() if (
            self.namespace + '/' in x.name)]
        return {_remap_name(x.name, self.namespace) : x for x in detector_variables}
    
    @property
    def weights(self):
        return self.variables
    