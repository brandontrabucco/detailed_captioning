'''Author: Brandon Trabucco, Copyright 2019
Test the image captioning model with some fake inputs.'''


import itertools
import tensorflow as tf
import numpy as np
from detailed_captioning.layers.image_captioner import ImageCaptioner
from detailed_captioning.utils import check_runtime
from detailed_captioning.utils import load_glove
from detailed_captioning.utils import load_image_from_path
from detailed_captioning.utils import get_object_detector_config
from detailed_captioning.utils import get_object_detector_checkpoint
from detailed_captioning.utils import get_image_captioner_checkpoint
from detailed_captioning.inputs.mscoco import import_mscoco


PRINT_STRING = """Training iteration {0} loss was {1} caption was {2}"""


if __name__ == "__main__":
    
    vocab, _ = load_glove(vocab_size=100000, embedding_size=300)
    
    with tf.device("/device:GPU:0"):

        with tf.Graph().as_default():

            image_tensor, image_ids, input_tensor, label_tensor, input_mask = (
                import_mscoco(is_training=True, batch_size=1, num_epochs=1))
            captioner = ImageCaptioner(get_object_detector_config(), 300, batch_size=1, 
                                       beam_size=3, vocab_size=100000, embedding_size=300)
            logits, ids = captioner(image_tensor, seq_inputs=input_tensor, 
                                    lengths=tf.reduce_sum(input_mask, axis=1))

            tf.losses.sparse_softmax_cross_entropy(label_tensor, logits, weights=input_mask)
            loss = tf.losses.get_total_loss()
            learning_step = tf.train.GradientDescentOptimizer(1.0).minimize(
                loss, var_list=captioner.variables.captioner_variables)

            captioner_saver = tf.train.Saver(var_list=captioner.variables.join())
            object_detector_saver = tf.train.Saver(var_list=captioner.variables.detector_variables)
            captioner_ckpt = get_image_captioner_checkpoint()
            object_detector_ckpt = get_object_detector_checkpoint()

            with tf.Session() as sess:

                if captioner_ckpt is not None:
                    captioner_saver.restore(sess, captioner_ckpt)
                else:
                    object_detector_saver.restore(sess, object_detector_ckpt)
                    sess.run(tf.variables_initializer(
                        list(captioner.variables.captioner_variables.values())))

                for i in itertools.count():
                    
                    try:
                        a, b, c = sess.run([ids, loss, learning_step])
                        print(PRINT_STRING.format(i, b, " ".join([
                            vocab.id_to_word(x) for x in a[0, :].tolist()])))
                        
                    except Exception as e:
                        print(e)
                        break

                    captioner_saver.save(sess, 'ckpts/caption_model/model.ckpt')
                    