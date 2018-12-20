'''Author: Brandon Trabucco, Copyright 2019
Test the image captioning model with some fake inputs.'''


import os
import time
import json
import itertools
import tensorflow as tf
import numpy as np
from detailed_captioning.layers.image_captioner import ImageCaptioner
from detailed_captioning.cells.show_and_tell_cell import ShowAndTellCell
from detailed_captioning.utils import check_runtime
from detailed_captioning.utils import load_glove
from detailed_captioning.utils import load_tagger
from detailed_captioning.utils import load_image_from_path
from detailed_captioning.utils import get_resnet_v2_101_checkpoint
from detailed_captioning.utils import get_show_and_tell_checkpoint
from detailed_captioning.utils import remap_decoder_name_scope
from detailed_captioning.utils import list_of_ids_to_string
from detailed_captioning.utils import recursive_ids_to_string
from detailed_captioning.utils import coco_get_metrics
from detailed_captioning.utils import get_train_annotations_file
from detailed_captioning.inputs.mean_image_features_only import import_mscoco
from glove.heuristic import get_descriptive_scores


PRINT_STRING = """({4:.2f} img/sec) iteration: {0:05d}\n    before: {1}\n    after: {2}\n    label: {3}"""
BATCH_SIZE = 32
BEAM_SIZE = 3
STYLE_UPDATES = 10


if __name__ == "__main__":
    
    vocab, pretrained_matrix = load_glove(vocab_size=100000, embedding_size=300)
    tagger = load_tagger()
    with tf.Graph().as_default():

        image_id, mean_features, input_seq, target_seq, indicator = (
            import_mscoco(mode="eval", batch_size=BATCH_SIZE, num_epochs=1, is_mini=True))
        image_captioner = ImageCaptioner(ShowAndTellCell(300), vocab, pretrained_matrix, 
            trainable=False, beam_size=BEAM_SIZE)
        
        # Save the image features so a gradient can be taken
        mean_image_features = tf.get_variable(
            "mean_image_features", dtype=tf.float32, shape=mean_features.shape)
        logits, ids = image_captioner(mean_image_features=mean_image_features)
        load_batch_op = tf.assign(mean_image_features, mean_features)
        # The words are sorted by frequency
        descriptive_scores = tf.constant(get_descriptive_scores(vocab.reverse_vocab, vocab, tagger))
        # Identify the relative frequency per word
        style_scores = tf.nn.embedding_lookup(descriptive_scores, ids)
        # Reward less frequent words
        tf.losses.sparse_softmax_cross_entropy(ids, logits, weights=style_scores)
        style_loss = tf.losses.get_total_loss()
        # Perform a policy gradient update
        descend_style_loss_op = tf.train.GradientDescentOptimizer(100.0).minimize(style_loss, 
            var_list=mean_image_features)
        
        captioner_saver = tf.train.Saver(var_list=remap_decoder_name_scope(image_captioner.variables))
        captioner_ckpt, captioner_ckpt_name = get_show_and_tell_checkpoint()

        with tf.Session() as sess:

            assert(captioner_ckpt is not None)
            captioner_saver.restore(sess, captioner_ckpt)
            used_ids = set()
            json_dump = []

            for i in itertools.count():
                time_start = time.time()
                try:
                    _target_seq, _image_id, _ = sess.run([target_seq, image_id, load_batch_op])
                    _before_ids = sess.run(ids)
                    for s in range(STYLE_UPDATES):
                        sess.run(descend_style_loss_op)
                    _after_ids = sess.run(ids)
                except:
                    break
                before_captions = recursive_ids_to_string(_before_ids[:, 0, :].tolist(), vocab)
                after_captions = recursive_ids_to_string(_after_ids[:, 0, :].tolist(), vocab)
                the_labels = recursive_ids_to_string(_target_seq[:, :].tolist(), vocab)
                the_image_ids = _image_id.tolist()
                for j, x1, x2, y in zip(the_image_ids, before_captions, after_captions, the_labels):
                    if not j in used_ids:
                        used_ids.add(j)
                        json_dump.append({"image_id": j, "caption": x1})
                print(PRINT_STRING.format(i, before_captions[0], after_captions[0], the_labels[0], 
                    BATCH_SIZE / (time.time() - time_start))) 

            print("Finishing evaluating.")
