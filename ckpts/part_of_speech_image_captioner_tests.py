"""Author: Brandon Trabucco, Copyright 2019.
Tests for the Image Captioner Architecture."""


import tensorflow as tf
import numpy as np
from detailed_captioning.utils import load_glove
from detailed_captioning.utils import get_parts_of_speech
from detailed_captioning.utils import get_faster_rcnn_config
from detailed_captioning.utils import get_faster_rcnn_checkpoint
from detailed_captioning.utils import get_resnet_v2_101_checkpoint
from detailed_captioning.utils import load_image_from_path
from detailed_captioning.layers.box_extractor import BoxExtractor
from detailed_captioning.layers.feature_extractor import FeatureExtractor
from detailed_captioning.layers.part_of_speech_image_captioner import PartOfSpeechImageCaptioner
from detailed_captioning.cells.up_down_cell import UpDownCell


tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):

    image = load_image_from_path("images/image.jpg")[np.newaxis, ...]

    vocab, pretrained_matrix = load_glove(vocab_size=100, embedding_size=50)
    pos, pos_embeddings = get_parts_of_speech(), np.random.normal(0, 0.1, [15, 50])
    with tf.Graph().as_default():
        
        inputs = tf.placeholder(tf.float32, shape=image.shape)
        box_extractor = BoxExtractor(get_faster_rcnn_config(), top_k_boxes=16)
        boxes, scores, cropped_inputs = box_extractor(inputs)
        feature_extractor = FeatureExtractor()
        mean_image_features = tf.reduce_mean(feature_extractor(inputs), [1, 2])
        mean_object_features = tf.reshape(
            tf.reduce_mean(feature_extractor(cropped_inputs), [1, 2]), [1, 16, 2048])
        image_captioner = PartOfSpeechImageCaptioner(
            UpDownCell(50), vocab, pretrained_matrix,
            UpDownCell(50), UpDownCell(50), pos, pos_embeddings)
        pos_logits, pos_logits_ids, word_logits, word_logits_ids = image_captioner(
            mean_image_features=mean_image_features, 
            mean_object_features=mean_object_features)

        with tf.Session() as sess:
            
            box_saver = tf.train.Saver(var_list=box_extractor.variables)
            resnet_saver = tf.train.Saver(var_list=feature_extractor.variables)
            
            box_saver.restore(sess, get_faster_rcnn_checkpoint())
            resnet_saver.restore(sess, get_resnet_v2_101_checkpoint())
            sess.run(tf.variables_initializer(image_captioner.variables))
            
            results = sess.run([pos_logits, pos_logits_ids, word_logits, 
                word_logits_ids], feed_dict={inputs: image})

            assert(results[2].shape[0] == 1 and results[2].shape[1] == 3 and results[2].shape[3] == 100)
            tf.logging.info("Successfully passed test.")
        

if __name__ == "__main__":
    
    tf.app.run()