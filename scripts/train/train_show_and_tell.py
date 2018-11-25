'''Author: Brandon Trabucco, Copyright 2019
Test the image captioning model with some fake inputs.'''


import time
import itertools
import tensorflow as tf
import numpy as np
from detailed_captioning.layers.image_captioner import ImageCaptioner
from detailed_captioning.cells.show_and_tell_cell import ShowAndTellCell
from detailed_captioning.utils import load_glove
from detailed_captioning.utils import get_show_and_tell_checkpoint 
from detailed_captioning.utils import list_of_ids_to_string
from detailed_captioning.inputs.mean_image_features_only import import_mscoco


PRINT_STRING = """({3:.2f} img/sec) iteration: {0:05d} loss: {1:.5f}\n    caption: {2}"""
BATCH_SIZE = 100
INITIAL_LEARNING_RATE = 2.0
TRAINING_EXAMPLES = 5000
EPOCHS_PER_DECAY = 8
DECAY_RATE = 1.0


def main(unused_argv):
    
    vocab, pretrained_matrix = load_glove(vocab_size=100000, embedding_size=300)
    with tf.Graph().as_default():

        image_id, mean_features, input_seq, target_seq, indicator = (
            import_mscoco(mode="train", batch_size=BATCH_SIZE, num_epochs=100, is_mini=True))
        show_and_tell_cell = ShowAndTellCell(300)
        image_captioner = ImageCaptioner(show_and_tell_cell, vocab, pretrained_matrix)
        logits, ids = image_captioner(lengths=tf.reduce_sum(indicator, axis=1), 
            mean_image_features=mean_features, seq_inputs=input_seq)
        tf.losses.sparse_softmax_cross_entropy(target_seq, logits, weights=indicator)
        loss = tf.losses.get_total_loss()
        
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, 
            global_step, (TRAINING_EXAMPLES // BATCH_SIZE) * EPOCHS_PER_DECAY, 
            DECAY_RATE, staircase=True)
        learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, 
            var_list=image_captioner.variables, global_step=global_step)

        captioner_saver = tf.train.Saver(var_list=image_captioner.variables + [global_step])
        captioner_ckpt, captioner_ckpt_name = get_show_and_tell_checkpoint()
        with tf.Session() as sess:
            
            if captioner_ckpt is not None:
                captioner_saver.restore(sess, captioner_ckpt)
            else:
                sess.run(tf.variables_initializer(image_captioner.variables + [global_step]))
            captioner_saver.save(sess, captioner_ckpt_name, global_step=global_step)
            last_save = time.time()
            
            for i in itertools.count():
                
                time_start = time.time()
                try:
                    _ids, _loss, _learning_step = sess.run([ids, loss, learning_step])
                except:
                    break
                    
                iteration = sess.run(global_step)
                    
                print(PRINT_STRING.format(
                    iteration, _loss, list_of_ids_to_string(_ids[0, :].tolist(), vocab), 
                    BATCH_SIZE / (time.time() - time_start)))
                
                new_save = time.time()
                if new_save - last_save > 3600: # save the model every hour
                    captioner_saver.save(sess, captioner_ckpt_name, global_step=global_step)
                    last_save = new_save
                    
            captioner_saver.save(sess, captioner_ckpt_name, global_step=global_step)
            print("Finishing training.")
        

if __name__ == "__main__":
    
    tf.app.run()
                    