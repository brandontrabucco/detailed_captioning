"""Author: Brandon Trabucco, Copyright 2019.
Tests for the Up-Down Image Captioner Architecture."""


import tensorflow as tf
import numpy as np
from detailed_captioning.utils import get_faster_rcnn_config
from detailed_captioning.utils import get_faster_rcnn_checkpoint
from detailed_captioning.utils import get_resnet_v2_101_checkpoint
from detailed_captioning.utils import load_image_from_path
from detailed_captioning.layers.box_extractor import BoxExtractor
from detailed_captioning.layers.image_captioner import ImageCaptioner
import matplotlib.pyplot as plt
import matplotlib.patches as patches


tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    
    with tf.device("/cpu:0"):

        image = load_image_from_path("images/image.jpg")[np.newaxis, ...]
        
        g = tf.Graph()
        with g.as_default():
            box_extractor = BoxExtractor(get_faster_rcnn_config())
            inputs = tf.placeholder(tf.float32, shape=image.shape)
            boxes, scores, cropped_inputs = box_extractor(inputs)
            image_captioner = ImageCaptioner(512, batch_size=1, beam_size=3, vocab_size=100)
            logits, ids = image_captioner(inputs, boxes, cropped_images=cropped_inputs)
            
        with tf.Session(graph=g) as sess:
            box_saver = tf.train.Saver(var_list=box_extractor.variables)
            resnet_saver = tf.train.Saver(var_list=image_captioner.variables.feature_extractor)
            box_saver.restore(sess, get_faster_rcnn_checkpoint())
            resnet_saver.restore(sess, get_resnet_v2_101_checkpoint())
            sess.run(tf.variables_initializer(image_captioner.variables.up_down_cell))
            results = sess.run([logits, ids], feed_dict={inputs: image})
            
        assert(results[0].shape[0] == 1 and results[0].shape[1] == 3 and results[0].shape[3] == 100)
        tf.logging.info("Successfully passed test.")
        

if __name__ == "__main__":
    
    tf.app.run()