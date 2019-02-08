'''Author: Brandon Trabucco, Copyright 2019
Run the attribute detector on images passed through the command line.'''


import os
import time
import tensorflow as tf
import numpy as np
from detailed_captioning.layers.box_extractor import BoxExtractor
from detailed_captioning.layers.attribute_detector import AttributeDetector
from detailed_captioning.layers.feature_extractor import FeatureExtractor
from detailed_captioning.utils import check_runtime
from detailed_captioning.utils import get_visual_attributes
from detailed_captioning.utils import load_image_from_path
from detailed_captioning.utils import get_resnet_v2_101_checkpoint
from detailed_captioning.utils import get_attribute_detector_checkpoint
from detailed_captioning.utils import get_faster_rcnn_config
from detailed_captioning.utils import get_faster_rcnn_checkpoint


PRINT_STRING = """
({0:.2f} img/sec) image: {1:05d}
    filename: {2}
    attributes: {3}"""

tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_string("file_pattern", "image.jpg", "")
FLAGS = tf.flags.FLAGS


if __name__ == "__main__":
    
    attribute_map = get_visual_attributes()
    with tf.Graph().as_default():

        list_of_filenames = tf.gfile.Glob(FLAGS.file_pattern)
        list_of_images = [load_image_from_path(filename) for filename in list_of_filenames]
        box_extractor = BoxExtractor(get_faster_rcnn_config())
        attribute_detector = AttributeDetector(1000)
        feature_extractor = FeatureExtractor()
        
        inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        resized_inputs = tf.image.resize_images(inputs, [224, 224])
        boxes, scores, cropped_inputs = box_extractor(inputs)
        image_features = tf.reduce_mean(feature_extractor(resized_inputs), [1, 2])
        object_features = tf.reduce_mean(feature_extractor(cropped_inputs), [1, 2])
        batch_size = tf.shape(image_features)[0]
        num_boxes = tf.shape(object_features)[0] // batch_size
        depth = tf.shape(image_features)[1]
        object_features = tf.reshape(object_features, [batch_size, num_boxes, depth])
        logits, image_detections, object_detections = attribute_detector(image_features, object_features)
        
        saver = tf.train.Saver(var_list=box_extractor.variables)
        feature_extractor_saver = tf.train.Saver(var_list=feature_extractor.variables)
        attribute_detector_saver = tf.train.Saver(var_list=attribute_detector.variables)
        
        faster_rcnn_ckpt = get_faster_rcnn_checkpoint()
        feature_extractor_ckpt = get_resnet_v2_101_checkpoint()
        attribute_detector_ckpt, attribute_detector_ckpt_name = get_attribute_detector_checkpoint()

        with tf.Session() as sess:

            assert(feature_extractor_ckpt is not None and attribute_detector_ckpt is not None)
            saver.restore(sess, faster_rcnn_ckpt)
            feature_extractor_saver.restore(sess, feature_extractor_ckpt)
            attribute_detector_saver.restore(sess, attribute_detector_ckpt)
            
            for i, (name, image) in enumerate(zip(list_of_filenames, list_of_images)):
                
                time_start = time.time()
                image_attributes = sess.run(image_detections, feed_dict={inputs: image[np.newaxis, ...]})
                print(PRINT_STRING.format(
                    1 / (time.time() - time_start),
                    i,
                    name,
                    str(attribute_map.id_to_word(image_attributes[0, :].tolist()))
                ))
                
                object_attributes = sess.run(object_detections, feed_dict={inputs: image[np.newaxis, ...]})
                for j in range(object_attributes.shape[1]):
                    print(PRINT_STRING.format(
                        1 / (time.time() - time_start),
                        i,
                        name + " box{0}".format(j),
                        str(attribute_map.id_to_word(object_attributes[0, j, :].tolist()))
                    ))

            print("Finishing running attribute detector.")
