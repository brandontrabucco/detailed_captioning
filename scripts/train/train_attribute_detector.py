'''Author: Brandon Trabucco, Copyright 2019
Test the image captioning model with some fake inputs.'''


import time
import itertools
import tensorflow as tf
import numpy as np
from detailed_captioning.layers.attribute_detector import AttributeDetector
from detailed_captioning.utils import load_glove
from detailed_captioning.utils import get_attribute_detector_checkpoint 
from detailed_captioning.utils import get_visual_attributes
from detailed_captioning.inputs.mean_image_features_and_attributes_only import import_mscoco


PRINT_STRING = """
({0:.2f} img/sec) iteration: {1:05d} loss: {2:.5f}
    predicted: {3}
    correct: {4}"""

tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_integer("num_epochs", 100, "")
tf.flags.DEFINE_integer("batch_size", 32, "")
tf.flags.DEFINE_boolean("is_mini", False, "")
FLAGS = tf.flags.FLAGS


def main(unused_argv):
    
    attribute_map = get_visual_attributes()
    
    with tf.Graph().as_default():

        image_id, mean_features, attributes, input_seq, target_seq, indicator = import_mscoco(
            mode="train", batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs, is_mini=FLAGS.is_mini)
        attribute_detector = AttributeDetector(1000)
        logits, detections, = attribute_detector(mean_features)
        tf.losses.sigmoid_cross_entropy(attributes, logits)
        loss = tf.losses.get_total_loss()
        
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer()
        learning_step = optimizer.minimize(loss, var_list=attribute_detector.variables, 
            global_step=global_step)

        captioner_saver = tf.train.Saver(var_list=attribute_detector.variables + [global_step])
        captioner_ckpt, captioner_ckpt_name = get_attribute_detector_checkpoint()
        with tf.Session() as sess:
            
            sess.run(tf.variables_initializer(optimizer.variables()))
            if captioner_ckpt is not None:
                captioner_saver.restore(sess, captioner_ckpt)
            else:
                sess.run(tf.variables_initializer(attribute_detector.variables + [global_step]))
            captioner_saver.save(sess, captioner_ckpt_name, global_step=global_step)
            last_save = time.time()
            
            for i in itertools.count():
                
                time_start = time.time()
                try:
                    _attributes, _detections, _loss, _ = sess.run([
                        attributes, detections, loss, learning_step])
                except:
                    break
                    
                iteration = sess.run(global_step)
                ground_truth_ids = np.where(_attributes[0, :] > 0.5)
                    
                print(PRINT_STRING.format(
                    FLAGS.batch_size / (time.time() - time_start),
                    iteration, 
                    _loss, 
                    str(attribute_map.id_to_word(_detections[0, :].tolist())),
                    str(attribute_map.id_to_word(ground_truth_ids[0].tolist())),
                    ))
                
                new_save = time.time()
                if new_save - last_save > 3600: # save the model every hour
                    captioner_saver.save(sess, captioner_ckpt_name, global_step=global_step)
                    last_save = new_save
                    
            captioner_saver.save(sess, captioner_ckpt_name, global_step=global_step)
            print("Finishing training.")
        

if __name__ == "__main__":
    
    tf.app.run()
                    