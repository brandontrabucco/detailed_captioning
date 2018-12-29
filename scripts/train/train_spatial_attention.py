'''Author: Brandon Trabucco, Copyright 2019
Test the image captioning model with some fake inputs.'''


import time
import itertools
import tensorflow as tf
import numpy as np
from detailed_captioning.layers.image_captioner import ImageCaptioner
from detailed_captioning.cells.spatial_attention_cell import SpatialAttentionCell
from detailed_captioning.utils import load_glove
from detailed_captioning.utils import get_spatial_attention_checkpoint 
from detailed_captioning.utils import list_of_ids_to_string
from detailed_captioning.inputs.spatial_image_features_only import import_mscoco


PRINT_STRING = """
({4:.2f} img/sec) iteration: {0:05d} loss: {1:.5f}
    caption: {2}
    actual: {3}"""

tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_integer("num_epochs", 100, "")
tf.flags.DEFINE_integer("batch_size", 32, "")
tf.flags.DEFINE_boolean("is_mini", False, "")
FLAGS = tf.flags.FLAGS


def main(unused_argv):
    
    vocab, pretrained_matrix = load_glove(vocab_size=100000, embedding_size=300)
    with tf.Graph().as_default():
        
        image_id, spatial_features, input_seq, target_seq, indicator = import_mscoco(
            mode="train", batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs, is_mini=FLAGS.is_mini)
        image_captioner = ImageCaptioner(SpatialAttentionCell(300), vocab, pretrained_matrix)
        logits, ids = image_captioner(lengths=tf.reduce_sum(indicator, axis=1), 
            spatial_image_features=spatial_features, seq_inputs=input_seq)
        tf.losses.sparse_softmax_cross_entropy(target_seq, logits, weights=indicator)
        loss = tf.losses.get_total_loss()
        
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer()
        learning_step = optimizer.minimize(loss, var_list=image_captioner.variables, global_step=global_step)

        captioner_saver = tf.train.Saver(var_list=image_captioner.variables + [global_step])
        captioner_ckpt, captioner_ckpt_name = get_spatial_attention_checkpoint()
        with tf.Session() as sess:
            
            sess.run(tf.variables_initializer(optimizer.variables()))
            if captioner_ckpt is not None:
                captioner_saver.restore(sess, captioner_ckpt)
            else:
                sess.run(tf.variables_initializer(image_captioner.variables + [global_step]))
            captioner_saver.save(sess, captioner_ckpt_name, global_step=global_step)
            last_save = time.time()
            
            for i in itertools.count():
                
                time_start = time.time()
                try:
                    _target, _ids, _loss, _learning_step = sess.run([target_seq, ids, loss, learning_step])
                except:
                    break
                    
                iteration = sess.run(global_step)
                    
                print(PRINT_STRING.format(
                    iteration, _loss, 
                    list_of_ids_to_string(_ids[0, :].tolist(), vocab), 
                    list_of_ids_to_string(_target[0, :].tolist(), vocab), 
                    FLAGS.batch_size / (time.time() - time_start)))
                
                new_save = time.time()
                
                if new_save - last_save > 3600:
                    captioner_saver.save(sess, captioner_ckpt_name, global_step=global_step)
                    last_save = new_save

            captioner_saver.save(sess, captioner_ckpt_name, global_step=global_step)
            print("Finishing training.")


if __name__ == "__main__":
    
    tf.app.run()
                    