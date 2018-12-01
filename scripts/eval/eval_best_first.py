'''Author: Brandon Trabucco, Copyright 2019
Test the image captioning model with some fake inputs.'''


import os
import time
import json
import itertools
import tensorflow as tf
import numpy as np
from detailed_captioning.layers.best_first_module import BestFirstModule
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


PRINT_STRING = """({0:.2f} img/sec) iteration: {1:05d}\n    before insert: {2}\n    after model insert: {3}\n    after label insert: {4}"""
BATCH_SIZE = 50


if __name__ == "__main__":
    
    vocab, pretrained_matrix = load_glove(vocab_size=100000, embedding_size=300)
    with tf.Graph().as_default():

        image_id, running_ids, indicator, previous_id, next_id, pointer, image_features = (
            import_mscoco(mode="train", batch_size=BATCH_SIZE, num_epochs=1, is_mini=True))
        best_first_module = BestFirstModule(pretrained_matrix)
        pointer_logits, word_logits = best_first_module(
            image_features, running_ids, previous_id, indicators=indicator)
        ids = tf.argmax(word_logits, axis=-1, output_type=tf.int32)
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
                    _caption, _ids, _next_id, _model_pointer, _label_pointer, _image_id = sess.run([
                        running_ids, ids, next_id, pointer_ids, pointer, image_id])
                except:
                    break
                #current_caption = recursive_ids_to_string(_caption.tolist(), vocab)
                #model_insert = recursive_ids_to_string(_ids.tolist(), vocab)
                #label_insert = recursive_ids_to_string(_next_id.tolist(), vocab)
                current_caption = _caption.tolist()
                model_insert = _ids.tolist()
                label_insert = _next_id.tolist()
                model_pointer = _model_pointer.tolist()
                label_pointer = _label_pointer.tolist()
                the_image_ids = _image_id.tolist()
                for cap, mins, lins, mptr, lptr, iid in zip(current_caption, model_insert, 
                                   label_insert, model_pointer, 
                                   label_pointer, the_image_ids):
                    model_cap = cap.copy()
                    model_cap.insert(mptr, mins)
                    label_cap = cap.copy()
                    label_cap.insert(lptr, lins)
                    cap = recursive_ids_to_string(cap, vocab)
                    model_cap = recursive_ids_to_string(model_cap, vocab)
                    label_cap = recursive_ids_to_string(label_cap, vocab)
                    if not iid in used_ids:
                        used_ids.add(iid)
                        json_dump.append({"image_id": iid, "caption": model_cap})
                    print(PRINT_STRING.format(
                        BATCH_SIZE / (time.time() - time_start), i, 
                        cap, 
                        model_cap, 
                        label_cap)) 

            print("Finishing evaluating.")
            coco_get_metrics(json_dump, "ckpts/show_and_tell/", get_train_annotations_file())
            