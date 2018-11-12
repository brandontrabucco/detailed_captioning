'''Author: Brandon Trabucco, Copyright 2019
Keras Layer API wrapper for training a TF Object Detector model.
Typically uses Faster R-CNN https://arxiv.org/abs/1506.01497.'''


import tensorflow as tf
from object_detection.utils.config_util import get_configs_from_pipeline_file
from object_detection.builders import model_builder


def _repeat_elements(num_elements, num_repeats):
    # Tensor: [[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, ...]]
    return tf.reshape(tf.tile(tf.expand_dims(tf.range(num_elements), 1), 
                              [1, num_repeats]), [-1])


class BoxExtractor(tf.keras.layers.Layer):
    '''Implments a mechanism for extracting bounding boxes.
    Typically uses Faster R-CNN https://arxiv.org/abs/1506.01497.'''

    def __init__(self, pipeline_config_file, top_k_boxes=8, name=None, trainable=False, **kwargs):
    
        super(BoxExtractor, self).__init__(trainable=trainable, **kwargs)
        pipeline_config = get_configs_from_pipeline_file(pipeline_config_file)
        self.top_k_boxes = top_k_boxes
        self.detection_model = model_builder.build(
            pipeline_config['model'], is_training=trainable)
    
    def __call__(self, inputs):
        
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        # Run the original image though the CNN and ROI detector
        preprocessed_inputs, true_shapes = self.detection_model.preprocess(inputs)
        features_dict = self.detection_model.predict(preprocessed_inputs, true_shapes)
        detector_dict = self.detection_model.postprocess(features_dict, true_shapes)
        self._trainable_variables = tf.trainable_variables()
        self._variables = tf.global_variables()
        # Collect the feature and ROI outputs
        boxes = detector_dict['detection_boxes']
        scores = detector_dict['detection_scores']
        # Use the top k scored boxes
        scores, top_k = tf.nn.top_k(scores, k=self.top_k_boxes)
        batch_ids = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, self.top_k_boxes])
        boxes = tf.gather_nd(boxes, tf.stack([batch_ids, top_k], 2))
        num_boxes = tf.shape(boxes)[1]
        # Crop inputs according to the ROI bounding boxes
        cropped_inputs = tf.image.crop_and_resize(inputs, tf.reshape(boxes, [-1, 4]), 
            _repeat_elements(batch_size, num_boxes), [height, width])
        return boxes, scores, cropped_inputs
        
    @property
    def trainable_variables(self):
        # Handles to the models internal shared trainable variables
        return self._trainable_variables
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        # Handles to the models internal shared global variables
        return self._variables
    
    @property
    def weights(self):
        return self.variables
    