'''Author: Brandon Trabucco, Copyright 2019
Test the image captioning model with some fake inputs.'''


import time
import itertools
import tensorflow as tf
import numpy as np
from detailed_captioning.layers.image_captioner import ImageCaptioner
from detailed_captioning.cells.visual_sentinel_cell import VisualSentinelCell
from detailed_captioning.utils import load_glove
from detailed_captioning.utils import get_visual_sentinel_checkpoint 
from detailed_captioning.utils import list_of_ids_to_string
from detailed_captioning.inputs.spatial_image_features_only import import_mscoco


PRINT_STRING = """({3:.2f} img/sec) iteration: {0:05d} loss: {1:.5f}\n    caption: {2}"""
BATCH_SIZE = 80


def main(unused_argv):
    
    vocab, pretrained_matrix = load_glove(vocab_size=100000, embedding_size=300)
    with tf.Graph().as_default():

        image_id, spatial_features, input_seq, target_seq, indicator = (
            import_mscoco(mode="train", batch_size=BATCH_SIZE, num_epochs=100, is_mini=True))
        visual_sentinel_cell = VisualSentinelCell(300)
        image_captioner = ImageCaptioner(visual_sentinel_cell, vocab, pretrained_matrix)
        logits, ids = image_captioner(lengths=tf.reduce_sum(indicator, axis=1), 
            spatial_image_features=spatial_features, seq_inputs=input_seq)
        tf.losses.sparse_softmax_cross_entropy(target_seq, logits, weights=indicator)
        loss = tf.losses.get_total_loss()
        learning_step = tf.train.GradientDescentOptimizer(1.0).minimize(loss, 
            var_list=image_captioner.variables)

        captioner_saver = tf.train.Saver(var_list=image_captioner.variables)
        captioner_ckpt, captioner_ckpt_name = get_visual_sentinel_checkpoint()
        with tf.Session() as sess:
            sess.run(tf.variables_initializer(image_captioner.variables))
            if captioner_ckpt is not None:
                captioner_saver.restore(sess, captioner_ckpt)
            captioner_saver.save(sess, captioner_ckpt_name)
            last_save = time.time()
            for i in itertools.count():
                time_start = time.time()
                try:
                    _ids, _loss, _learning_step = sess.run([ids, loss, learning_step])
                except:
                    break
                print(PRINT_STRING.format(
                    i, _loss, list_of_ids_to_string(_ids[0, :].tolist(), vocab), 
                    BATCH_SIZE / (time.time() - time_start)))
                new_save = time.time()
                if new_save - last_save > 3600: # save the model every hour
                    captioner_saver.save(sess, captioner_ckpt_name)
                    last_save = new_save
                    
            captioner_saver.save(sess, captioner_ckpt_name)
            print("Finishing training.")
        

if __name__ == "__main__":
    
    tf.app.run()
                    