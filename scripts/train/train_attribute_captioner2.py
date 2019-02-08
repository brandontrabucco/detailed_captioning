'''Author: Brandon Trabucco, Copyright 2019
Test the attribute captioning model with COCO dataset inputs.'''


import time
import itertools
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from detailed_captioning.utils import load_glove
from detailed_captioning.layers.attribute_detector import AttributeDetector
from detailed_captioning.utils import get_attribute_detector_checkpoint 
from detailed_captioning.utils import get_visual_attributes
from detailed_captioning.layers.feature_extractor import FeatureExtractor
from detailed_captioning.utils import get_resnet_v2_101_checkpoint
from detailed_captioning.layers.attribute_captioner import AttributeCaptioner
from detailed_captioning.cells.grounded_attribute_cell import GroundedAttributeCell
from detailed_captioning.utils import get_grounded_attribute_checkpoint 
from detailed_captioning.utils import list_of_ids_to_string
from detailed_captioning.inputs.image_and_objects_only import import_mscoco


PRINT_STRING = """
({4:.2f} img/sec) iteration: {0:05d} loss: {1:.5f}
    caption: {2}
    actual: {3}"""

tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_integer("num_epochs", 100, "")
tf.flags.DEFINE_integer("batch_size", 1, "")
FLAGS = tf.flags.FLAGS


def main(unused_argv):
    
    #vocab, pretrained_matrix = load_glove(vocab_size=100000, embedding_size=300)
    #attribute_map = get_visual_attributes()
    #attribute_to_word_lookup_table = vocab.word_to_id(attribute_map.reverse_vocab)
    
    with tf.Graph().as_default():

        (image_id, image, cropped_image, input_seq, target_seq, indicator
            ) = import_mscoco(
                mode="train", batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)
        
        with tf.Session() as sess:
            
            image = sess.run(cropped_image)[0, :, :, :]
            plt.imshow(image)
            plt.savefig("test.png")
        

if __name__ == "__main__":
    
    tf.app.run()
                    