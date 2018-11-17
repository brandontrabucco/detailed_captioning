'''Author: Brandon Trabucco, Copyright 2019
Test the image captioning model with some fake inputs.'''


import time
import itertools
import tensorflow as tf
import numpy as np
from detailed_captioning.layers.image_captioner import ImageCaptioner
from detailed_captioning.cells.up_down_cell import UpDownCell
from detailed_captioning.utils import load_glove
from detailed_captioning.utils import get_up_down_checkpoint 
from detailed_captioning.utils import list_of_ids_to_string
from detailed_captioning.inputs.mscoco import import_mscoco


PRINT_STRING = """Training iteration {0:07d} loss was {1:.7f} caption was {2}"""


def main(unused_argv):
    
    vocab, pretrained_matrix = load_glove(vocab_size=100000, embedding_size=300)
    with tf.Graph().as_default():

        image_id, image, spatial_features, object_features, input_seq, target_seq, indicator = (
            import_mscoco(mode="train", batch_size=32, num_epochs=100, is_mini=True))
        up_down_cell = UpDownCell(300)
        image_captioner = ImageCaptioner(up_down_cell, vocab, pretrained_matrix)
        logits, ids = image_captioner(lengths=tf.reduce_sum(indicator, axis=1), 
            mean_image_features=tf.reduce_mean(spatial_features, [1, 2]),
            mean_object_features=object_features, seq_inputs=input_seq)
        tf.losses.sparse_softmax_cross_entropy(target_seq, logits, weights=indicator)
        loss = tf.losses.get_total_loss()
        learning_step = tf.train.GradientDescentOptimizer(1.0).minimize(loss, 
            var_list=image_captioner.variables)

        captioner_saver = tf.train.Saver(var_list=image_captioner.variables)
        captioner_ckpt, captioner_ckpt_name = get_up_down_checkpoint()
        with tf.Session() as sess:
            sess.run(tf.variables_initializer(image_captioner.variables))
            if captioner_ckpt is not None:
                captioner_saver.restore(sess, captioner_ckpt)
            captioner_saver.save(sess, captioner_ckpt_name)
            last_save = time.time()
            for i in itertools.count():
                try:
                    _ids, _loss, _learning_step = sess.run([ids, loss, learning_step])
                except:
                    break
                print(PRINT_STRING.format(i, _loss, list_of_ids_to_string(_ids[0, :].tolist(), vocab)))
                new_save = time.time()
                if new_save - last_save > 3600: # save the model every hour
                    captioner_saver.save(sess, captioner_ckpt_name)
                    last_save = new_save
                    
            print("Finishing training.")
        

if __name__ == "__main__":
    
    tf.app.run()
                    