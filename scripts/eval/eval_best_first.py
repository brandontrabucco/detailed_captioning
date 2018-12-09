'''Author: Brandon Trabucco, Copyright 2019
Test the image captioning model with some fake inputs.'''


import os
import time
import json
import itertools
import tensorflow as tf
import numpy as np
from detailed_captioning.layers.best_first_module import BestFirstModule
from detailed_captioning.cells.show_and_tell_cell import ShowAndTellCell
from detailed_captioning.utils import check_runtime
from detailed_captioning.utils import load_glove
from detailed_captioning.utils import load_image_from_path
from detailed_captioning.utils import get_resnet_v2_101_checkpoint
from detailed_captioning.utils import get_best_first_checkpoint
from detailed_captioning.utils import remap_decoder_name_scope
from detailed_captioning.utils import list_of_ids_to_string
from detailed_captioning.utils import recursive_ids_to_string
from detailed_captioning.utils import coco_get_metrics
from detailed_captioning.utils import get_train_annotations_file
from detailed_captioning.inputs.mean_image_features_best_first_only import import_mscoco


PRINT_STRING = """({0:.2f} img/sec) iteration: {1:05d}\n    caption: {2}"""
BATCH_SIZE = 50


if __name__ == "__main__":
    
    vocab, pretrained_matrix = load_glove(vocab_size=100000, embedding_size=300)
    with tf.Graph().as_default():

        _image_id, _running_ids, _indicator, _next_id, _pointer, _image_features = (
            import_mscoco(mode="train", batch_size=BATCH_SIZE, num_epochs=1, is_mini=True))
        
        image_id = tf.get_variable("image_id", dtype=_image_id.dtype, shape=_image_id.shape)
        image_features = tf.get_variable(
            "image_features", dtype=_image_features.dtype, shape=_image_features.shape)
        load_batch = tf.group(tf.assign(image_id, _image_id), tf.assign(image_features, _image_features))
        running_ids = tf.placeholder(tf.int32, name="running_ids", shape=[None, None])
        
        best_first_module = BestFirstModule(ShowAndTellCell(300), vocab, pretrained_matrix)
        pointer_logits, word_logits = best_first_module(running_ids, mean_image_features=image_features)
        word_ids = tf.argmax(word_logits, axis=-1, output_type=tf.int32)
        pointer_ids = tf.argmax(pointer_logits, axis=-1, output_type=tf.int32)
        
        captioner_saver = tf.train.Saver(var_list=remap_decoder_name_scope(best_first_module.variables))
        captioner_ckpt, captioner_ckpt_name = get_best_first_checkpoint()

        with tf.Session() as sess:

            assert(captioner_ckpt is not None)
            captioner_saver.restore(sess, captioner_ckpt)
            used_ids = set()
            json_dump = []

            for i in itertools.count():
                time_start = time.time()
                try:

                    sess.run(load_batch)
                    current_image_id = sess.run(image_id).tolist()
                    closed = [False] * BATCH_SIZE
                    current_running_ids = [[vocab.start_id, vocab.end_id]] * BATCH_SIZE
                    i = 0
                    while not all(closed) and i < 20:
                        i = i + 1
                        _word_ids, _pointer_ids = sess.run([word_ids, pointer_ids], feed_dict={
                            "running_ids:0": current_running_ids})
                        _word_ids, _pointer_ids = _word_ids.tolist(), _pointer_ids.tolist()
                        for i in range(BATCH_SIZE):
                            if closed[i] or (
                                    _word_ids[i] == vocab.end_id or 
                                    _pointer_ids[i] = len(current_running_ids[i] - 1)):
                                closed[i] = True
                                continue
                            current_running_ids[i].insert(_pointer_ids[i] + 1, _word_ids[i])

                except:
                    break
                    
                for i in range(BATCH_SIZE):
                    iid = current_image_id[i]
                    cap = recursive_ids_to_string(current_running_ids[i][1:-1], vocab)
                    
                    if not iid in used_ids:
                        used_ids.add(iid)
                        json_dump.append({"image_id": iid, "caption": cap})
                        
                    print(PRINT_STRING.format(
                        BATCH_SIZE / (time.time() - time_start), i, cap)) 

            print("Finishing evaluating.")
            coco_get_metrics(json_dump, "ckpts/best_first/", get_train_annotations_file())
            