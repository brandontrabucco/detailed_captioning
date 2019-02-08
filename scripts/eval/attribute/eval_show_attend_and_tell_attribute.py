'''Author: Brandon Trabucco, Copyright 2019
Test the image captioning model with some fake inputs.'''


import os
import time
import json
import itertools
import tensorflow as tf
import numpy as np
from detailed_captioning.layers.attribute_detector import AttributeDetector
from detailed_captioning.layers.attribute_image_captioner import AttributeImageCaptioner
from detailed_captioning.cells.show_attend_and_tell_cell import ShowAttendAndTellCell
from detailed_captioning.utils import check_runtime
from detailed_captioning.utils import load_glove
from detailed_captioning.utils import load_image_from_path
from detailed_captioning.utils import get_visual_attributes
from detailed_captioning.utils import get_show_attend_and_tell_attribute_checkpoint 
from detailed_captioning.utils import get_attribute_detector_checkpoint 
from detailed_captioning.utils import remap_decoder_name_scope
from detailed_captioning.utils import list_of_ids_to_string
from detailed_captioning.utils import recursive_ids_to_string
from coco_metrics import evaluate
from detailed_captioning.utils import get_train_annotations_file
from detailed_captioning.utils import get_val_annotations_file
from detailed_captioning.inputs.spatial_image_features_only import import_mscoco


PRINT_STRING = """({3:.2f} img/sec) iteration: {0:05d}\n    caption: {1}\n    label: {2}"""
tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_integer("batch_size", 1, "")
tf.flags.DEFINE_integer("beam_size", 3, "")
tf.flags.DEFINE_boolean("is_mini", False, "")
tf.flags.DEFINE_string("mode", "eval", "")
FLAGS = tf.flags.FLAGS


if __name__ == "__main__":
    
    vocab, pretrained_matrix = load_glove(vocab_size=100000, embedding_size=300)
    attribute_map, attribute_embeddings_map = get_visual_attributes(), np.random.normal(0, 0.1, [1000, 2048])
    with tf.Graph().as_default():

        image_id, spatial_features, input_seq, target_seq, indicator = import_mscoco(
            mode=FLAGS.mode, batch_size=FLAGS.batch_size, num_epochs=1, is_mini=FLAGS.is_mini)
        show_attend_and_tell_cell = ShowAttendAndTellCell(300, num_image_features=2048)
        attribute_image_captioner = AttributeImageCaptioner(
            show_attend_and_tell_cell, vocab, pretrained_matrix,
            attribute_map, attribute_embeddings_map)
        attribute_detector = AttributeDetector(1000)
        _, top_k_attributes = attribute_detector(tf.reduce_mean(spatial_features, [1, 2]))
        logits, ids = attribute_image_captioner(top_k_attributes,
            spatial_image_features=spatial_features)

        captioner_saver = tf.train.Saver(var_list=remap_decoder_name_scope(
            attribute_image_captioner.variables))
        attribute_detector_saver = tf.train.Saver(var_list=attribute_detector.variables)
        captioner_ckpt, captioner_ckpt_name = get_show_attend_and_tell_attribute_checkpoint()
        attribute_detector_ckpt, attribute_detector_ckpt_name = get_attribute_detector_checkpoint()

        with tf.Session() as sess:

            assert(captioner_ckpt is not None and attribute_detector_ckpt is not None)
            captioner_saver.restore(sess, captioner_ckpt)
            attribute_detector_saver.restore(sess, attribute_detector_ckpt)
            used_ids = set()
            json_dump = []

            for i in itertools.count():
                time_start = time.time()
                try:
                    _ids, _target_seq, _image_id = sess.run([ids, target_seq, image_id])
                except:
                    break
                the_captions = recursive_ids_to_string(_ids[:, 0, :].tolist(), vocab)
                the_labels = recursive_ids_to_string(_target_seq[:, :].tolist(), vocab)
                the_image_ids = _image_id.tolist()
                for j, x, y in zip(the_image_ids, the_captions, the_labels):
                    if not j in used_ids:
                        used_ids.add(j)
                        json_dump.append({"image_id": j, "caption": x})
                print(PRINT_STRING.format(i, the_captions[0], the_labels[0], 
                    FLAGS.batch_size / (time.time() - time_start))) 

            print("Finishing evaluating.")
            evaluate(
                FLAGS.mode,
                json_dump, 
                captioner_ckpt_name.replace("model.ckpt", ""), 
                (get_train_annotations_file() if FLAGS.mode in ["train", "eval"] 
                    else get_val_annotations_file()))
