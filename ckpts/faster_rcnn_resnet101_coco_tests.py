"""Author: Brandon Trabucco, Copyright 2019.
Tests for the Resnet-101 CNN Architecture."""


import tensorflow as tf
import numpy as np
from detailed_captioning.utils import get_faster_rcnn_config
from detailed_captioning.utils import get_faster_rcnn_checkpoint
from detailed_captioning.utils import load_image_from_path
from detailed_captioning.layers.box_extractor import BoxExtractor
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
            
        with tf.Session(graph=g) as sess:
            saver = tf.train.Saver(var_list=box_extractor.variables)
            saver.restore(sess, get_faster_rcnn_checkpoint())
            results = sess.run([boxes, scores, cropped_inputs], feed_dict={inputs: image})
            
        assert(all([x == y for x, y in zip(results[0].shape, [1, 100, 4])]))
        tf.logging.info("Successfully passed test.")
        
        height = image.shape[1]
        width = image.shape[2]
        fig, ax = plt.subplots(1)
        ax.imshow(image[0, :])
        
        for i in range(results[0].shape[1]):
            
            this_box = results[0][0, i, :]
            box_y1 = this_box[0] * height
            box_x1 = this_box[1] * width
            box_y2 = this_box[2] * height
            box_x2 = this_box[3] * width
            
            rect = patches.Rectangle(
                (box_x1, box_y1),
                (box_x2 - box_x1),
                (box_y2 - box_y1),
                linewidth=1, edgecolor='r', facecolor='none')
            
            ax.add_patch(rect)
            
        plt.savefig("images/image_boxes.png")
        

if __name__ == "__main__":
    
    tf.app.run()