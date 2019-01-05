'''Author: Brandon Trabucco, Copyright 2019
Run the attribute detector on images passed through the command line.'''


import os
import time
import tensorflow as tf
import numpy as np
from detailed_captioning.layers.attribute_detector import AttributeDetector
from detailed_captioning.layers.feature_extractor import FeatureExtractor
from detailed_captioning.utils import check_runtime
from detailed_captioning.utils import get_visual_attributes
from detailed_captioning.utils import load_image_from_path
from detailed_captioning.utils import get_resnet_v2_101_checkpoint
from detailed_captioning.utils import get_attribute_detector_checkpoint


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
        attribute_detector = AttributeDetector(1000)
        feature_extractor = FeatureExtractor()
        
        inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        resized_inputs = tf.image.resize_images(inputs, [224, 224])
        features = tf.reduce_mean(feature_extractor(resized_inputs), [1, 2])
        _, attributes = attribute_detector(features)
        
        feature_extractor_saver = tf.train.Saver(var_list=feature_extractor.variables)
        attribute_detector_saver = tf.train.Saver(var_list=attribute_detector.variables)
        feature_extractor_ckpt = get_resnet_v2_101_checkpoint()
        attribute_detector_ckpt, attribute_detector_ckpt_name = get_attribute_detector_checkpoint()

        with tf.Session() as sess:

            assert(feature_extractor_ckpt is not None and attribute_detector_ckpt is not None)
            feature_extractor_saver.restore(sess, feature_extractor_ckpt)
            attribute_detector_saver.restore(sess, attribute_detector_ckpt)
            
            for i, (name, image) in enumerate(zip(list_of_filenames, list_of_images)):
                
                time_start = time.time()
                np_attributes = sess.run(attributes, feed_dict={inputs: image[np.newaxis, ...]})
                print(PRINT_STRING.format(
                    1 / (time.time() - time_start),
                    i,
                    name,
                    str(attribute_map.id_to_word(np_attributes[0, :].tolist()))
                ))

            print("Finishing running attribute detector.")
