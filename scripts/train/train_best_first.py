'''Author: Brandon Trabucco, Copyright 2019
Test the image captioning model with some fake inputs.'''


import time
import itertools
import tensorflow as tf
import numpy as np
from detailed_captioning.layers.best_first_image_captioner import BestFirstImageCaptioner
from detailed_captioning.cells.show_and_tell_cell import ShowAndTellCell
from detailed_captioning.utils import load_glove
from detailed_captioning.utils import get_best_first_checkpoint 
from detailed_captioning.utils import list_of_ids_to_string
from detailed_captioning.inputs.mean_image_features_best_first_only import import_mscoco


PRINT_STRING = """
({4:.2f} img/sec) iteration: {0:05d} loss: {1:.5f}
    caption: {2}
    actual: {3}"""

tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_integer("num_epochs", 100, "")
tf.flags.DEFINE_integer("batch_size", 32, "")
tf.flags.DEFINE_boolean("is_mini", False, "")
FLAGS = tf.flags.FLAGS


def insertion_sequence_to_array(wids, pids, lengths, vocab):
    out_array = []
    for i in range(wids.shape[0]):
        temp_array = [vocab.start_id, vocab.end_id]
        for j in range(wids.shape[1]):
            if j >= lengths[i] or pids[i, j] + 1 >= len(temp_array):
                break
            temp_array.insert(wids[i, j], pids[i, j] + 1)
        out_array.append(temp_array)
    return out_array
        


def main(unused_argv):
    
    vocab, pretrained_matrix = load_glove(vocab_size=100000, embedding_size=300)
    with tf.Graph().as_default():

        image_id, image_features, indicator, word_ids, pointer_ids = import_mscoco(
            mode="train", batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs, is_mini=FLAGS.is_mini)
        lengths = tf.reduce_sum(indicator, [1])
        show_and_tell_cell = ShowAndTellCell(300)
        best_first_image_captioner = BestFirstImageCaptioner(show_and_tell_cell, vocab, pretrained_matrix)
        word_logits, wids, pointer_logits, pids, ids, _lengths = best_first_image_captioner(
            mean_image_features=image_features,
            word_ids=word_ids, pointer_ids=pointer_ids, lengths=lengths)
        tf.losses.sparse_softmax_cross_entropy(pointer_ids, pointer_logits)
        tf.losses.sparse_softmax_cross_entropy(word_ids, word_logits)
        loss = tf.losses.get_total_loss()
        
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer()
        learning_step = optimizer.minimize(loss, var_list=best_first_image_captioner.variables, 
            global_step=global_step)

        captioner_saver = tf.train.Saver(var_list=best_first_image_captioner.variables + [global_step])
        captioner_ckpt, captioner_ckpt_name = get_best_first_checkpoint()
        with tf.Session() as sess:
            
            sess.run(tf.variables_initializer(optimizer.variables()))
            if captioner_ckpt is not None:
                captioner_saver.restore(sess, captioner_ckpt)
            else:
                sess.run(tf.variables_initializer(best_first_image_captioner.variables + [global_step]))
            captioner_saver.save(sess, captioner_ckpt_name, global_step=global_step)
            last_save = time.time()
            
            for i in itertools.count():
                
                time_start = time.time()
                try:
                    twids, tpids, _ids, _lengths, _loss, _learning_step = sess.run([
                        word_ids, pointer_ids, ids, lengths, loss, learning_step])
                except:
                    break
                    
                iteration = sess.run(global_step)
                
                insertion_sequence = insertion_sequence_to_array(twids, tpids, _lengths, vocab)
                    
                print(PRINT_STRING.format(
                    iteration, _loss, 
                    list_of_ids_to_string(insertion_sequence[0], vocab), 
                    list_of_ids_to_string(twids[0, :].tolist(), vocab), 
                    FLAGS.batch_size / (time.time() - time_start)))
                
                new_save = time.time()
                if new_save - last_save > 3600: # save the model every hour
                    captioner_saver.save(sess, captioner_ckpt_name, global_step=global_step)
                    last_save = new_save
                    
            captioner_saver.save(sess, captioner_ckpt_name, global_step=global_step)
            print("Finishing training.")
        

if __name__ == "__main__":
    
    tf.app.run()
                    