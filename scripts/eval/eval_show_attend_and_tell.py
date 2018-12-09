'''Author: Brandon Trabucco, Copyright 2019
Test the image captioning model with some fake inputs.'''


import os
import time
import json
import itertools
import tensorflow as tf
import numpy as np
from detailed_captioning.layers.image_captioner import ImageCaptioner
from detailed_captioning.cells.show_attend_and_tell_cell import ShowAttendAndTellCell
from detailed_captioning.utils import check_runtime
from detailed_captioning.utils import load_glove
from detailed_captioning.utils import load_image_from_path
from detailed_captioning.utils import get_resnet_v2_101_checkpoint
from detailed_captioning.utils import get_show_attend_and_tell_checkpoint
from detailed_captioning.utils import remap_decoder_name_scope
from detailed_captioning.utils import list_of_ids_to_string
from detailed_captioning.utils import recursive_ids_to_string
from detailed_captioning.utils import coco_get_metrics
from detailed_captioning.utils import get_train_annotations_file
from detailed_captioning.inputs.spatial_image_features_only import import_mscoco


PRINT_STRING = """({3:.2f} img/sec) iteration: {0:05d}\n    caption: {1}\n    label: {2}"""
tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_integer("batch_size", 1, "")
tf.flags.DEFINE_integer("beam_size", 3, "")
tf.flags.DEFINE_boolean("is_mini", False, "")
FLAGS = tf.flags.FLAGS


if __name__ == "__main__":
    
    vocab, pretrained_matrix = load_glove(vocab_size=100000, embedding_size=300)
    with tf.Graph().as_default():

        image_id, spatial_features, input_seq, target_seq, indicator = import_mscoco(
            mode="train", batch_size=FLAGS.batch_size, num_epochs=1, is_mini=FLAGS.is_mini)
        image_captioner = ImageCaptioner(ShowAttendAndTellCell(300), vocab, pretrained_matrix, 
            trainable=False, beam_size=FLAGS.beam_size)
        logits, ids = image_captioner(spatial_image_features=spatial_features)
        captioner_saver = tf.train.Saver(var_list=remap_decoder_name_scope(image_captioner.variables))
        captioner_ckpt, captioner_ckpt_name = get_show_attend_and_tell_checkpoint()

        with tf.Session() as sess:

            assert(captioner_ckpt is not None)
            captioner_saver.restore(sess, captioner_ckpt)
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
            coco_get_metrics(json_dump, "ckpts/show_attend_and_tell/", get_train_annotations_file())
