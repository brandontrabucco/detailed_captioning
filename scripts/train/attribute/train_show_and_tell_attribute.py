'''Author: Brandon Trabucco, Copyright 2019
Train an image caption model using attributes and image features.'''


import time
import itertools
import tensorflow as tf
import numpy as np
from detailed_captioning.layers.attribute_detector import AttributeDetector
from detailed_captioning.layers.attribute_image_captioner import AttributeImageCaptioner
from detailed_captioning.cells.show_and_tell_cell import ShowAndTellCell
from detailed_captioning.utils import load_glove
from detailed_captioning.utils import get_visual_attributes
from detailed_captioning.utils import get_show_and_tell_attribute_checkpoint 
from detailed_captioning.utils import get_attribute_detector_checkpoint 
from detailed_captioning.utils import list_of_ids_to_string
from detailed_captioning.inputs.mean_image_features_only import import_mscoco


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
    attribute_map, attribute_embeddings_map = get_visual_attributes(), np.random.normal(0, 0.1, [1000, 2048])
    with tf.Graph().as_default():

        image_id, mean_features, input_seq, target_seq, indicator = import_mscoco(
            mode="train", batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs, is_mini=FLAGS.is_mini)
        show_and_tell_cell = ShowAndTellCell(300, num_image_features=4096)
        attribute_image_captioner = AttributeImageCaptioner(
            show_and_tell_cell, vocab, pretrained_matrix,
            attribute_map, attribute_embeddings_map)
        attribute_detector = AttributeDetector(1000)
        _, top_k_attributes = attribute_detector(mean_features)
        logits, ids = attribute_image_captioner(top_k_attributes,
            lengths=tf.reduce_sum(indicator, axis=1), 
            mean_image_features=mean_features, 
            seq_inputs=input_seq)
        tf.losses.sparse_softmax_cross_entropy(target_seq, logits, weights=indicator)
        loss = tf.losses.get_total_loss()
        
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer()
        learning_step = optimizer.minimize(loss, var_list=attribute_image_captioner.variables, 
            global_step=global_step)

        captioner_saver = tf.train.Saver(var_list=attribute_image_captioner.variables + [global_step])
        attribute_detector_saver = tf.train.Saver(var_list=attribute_detector.variables)
        captioner_ckpt, captioner_ckpt_name = get_show_and_tell_attribute_checkpoint()
        attribute_detector_ckpt, attribute_detector_ckpt_name = get_attribute_detector_checkpoint()
        with tf.Session() as sess:
            
            sess.run(tf.variables_initializer(optimizer.variables()))
            if captioner_ckpt is not None:
                captioner_saver.restore(sess, captioner_ckpt)
            else:
                sess.run(tf.variables_initializer(attribute_image_captioner.variables + [global_step]))
            if attribute_detector_ckpt is not None:
                attribute_detector_saver.restore(sess, attribute_detector_ckpt)
            else:
                sess.run(tf.variables_initializer(attribute_detector.variables))
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
                if new_save - last_save > 3600: # save the model every hour
                    captioner_saver.save(sess, captioner_ckpt_name, global_step=global_step)
                    last_save = new_save
                    
            captioner_saver.save(sess, captioner_ckpt_name, global_step=global_step)
            print("Finishing training.")
        

if __name__ == "__main__":
    
    tf.app.run()
                    