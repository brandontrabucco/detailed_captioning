"""Author: Brandon Trabucco, Copyright 2019.
Tests for the Resnet-101 CNN Architecture."""


import tensorflow as tf
import numpy as np
from detailed_captioning.utils import get_resnet_v2_101_checkpoint
from detailed_captioning.utils import load_image_from_path
from detailed_captioning.layers.feature_extractor import FeatureExtractor


tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    
    with tf.device("/cpu:0"):

        image = np.random.uniform(-1, 1, [1, 224, 224, 3])
        
        g = tf.Graph()
        with g.as_default():
            feature_extractor = FeatureExtractor()
            inputs = tf.placeholder(tf.float32, shape=image.shape)
            features = feature_extractor(inputs)
            
        with tf.Session(graph=g) as sess:
            saver = tf.train.Saver(var_list=feature_extractor.trainable_variables)
            saver.restore(sess, get_resnet_v2_101_checkpoint())
            results = sess.run(features, feed_dict={inputs: image})
            
        assert(all([x == y for x, y in zip(results.shape, [1, 1, 1, 2048])]))
        tf.logging.info("Successfully passed test.")
        

if __name__ == "__main__":
    
    tf.app.run()
