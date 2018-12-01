'''Author: Brandon Trabucco, Copyright 2019
Test the image captioning model with some fake inputs.'''


import time
import itertools
import tensorflow as tf
import numpy as np
from detailed_captioning.layers.best_first_module import BestFirstModule
from detailed_captioning.utils import load_glove
from detailed_captioning.utils import get_best_first_checkpoint 
from detailed_captioning.utils import list_of_ids_to_string
from detailed_captioning.inputs.mean_image_features_best_first_only import import_mscoco


PRINT_STRING = """({3:.2f} img/sec) iteration: {0:05d} loss: {1:.5f}\n    caption: {2}"""
BATCH_SIZE = 32
LEARNING_RATE = 1.0


def main(unused_argv):
    
    vocab, pretrained_matrix = load_glove(vocab_size=100000, embedding_size=300)
    with tf.Graph().as_default():

        image_id, running_ids, indicator, previous_id, next_id, pointer, image_features = (
            import_mscoco(mode="train", batch_size=BATCH_SIZE, num_epochs=100, is_mini=True))
        best_first_module = BestFirstModule(pretrained_matrix)
        pointer_logits, word_logits = best_first_module(
            image_features, running_ids, previous_id, indicators=indicator, pointer_ids=pointer)
        tf.losses.sparse_softmax_cross_entropy(pointer, pointer_logits)
        tf.losses.sparse_softmax_cross_entropy(next_id, word_logits)
        loss = tf.losses.get_total_loss()
        
        ids = tf.argmax(word_logits, axis=-1, output_type=tf.int32)
        
        global_step = tf.train.get_or_create_global_step()
        learning_rate = LEARNING_RATE
        learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, 
            var_list=best_first_module.variables, global_step=global_step)

        captioner_saver = tf.train.Saver(var_list=best_first_module.variables + [global_step])
        captioner_ckpt, captioner_ckpt_name = get_best_first_checkpoint()
        with tf.Session() as sess:
            
            if captioner_ckpt is not None:
                captioner_saver.restore(sess, captioner_ckpt)
            else:
                sess.run(tf.variables_initializer(best_first_module.variables + [global_step]))
            captioner_saver.save(sess, captioner_ckpt_name, global_step=global_step)
            last_save = time.time()
            _ids, _loss, _learning_step = sess.run([ids, loss, learning_step])
            
            for i in itertools.count():
                
                time_start = time.time()
                try:
                    _ids, _loss, _learning_step = sess.run([ids, loss, learning_step])
                except:
                    break
                    
                iteration = sess.run(global_step)
                    
                print(PRINT_STRING.format(
                    iteration, _loss, list_of_ids_to_string(_ids.tolist(), vocab), 
                    BATCH_SIZE / (time.time() - time_start)))
                
                new_save = time.time()
                if new_save - last_save > 3600: # save the model every hour
                    captioner_saver.save(sess, captioner_ckpt_name, global_step=global_step)
                    last_save = new_save
                    
            captioner_saver.save(sess, captioner_ckpt_name, global_step=global_step)
            print("Finishing training.")
        

if __name__ == "__main__":
    
    tf.app.run()
                    