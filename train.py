'''Author: Brandon Trabucco, Copyright 2019
Test the image captioning model with some fake inputs.'''


import itertools
import tensorflow as tf
import numpy as np
from detailed_captioning.layers.image_captioner import ImageCaptioner
from detailed_captioning.utils import check_runtime
from detailed_captioning.utils import load_glove
from detailed_captioning.utils import load_image_from_path
from detailed_captioning.utils import get_resnet_v2_101_checkpoint
from detailed_captioning.utils import get_image_captioner_checkpoint
from detailed_captioning.inputs.mscoco import import_mscoco


PRINT_STRING = """Training iteration {0} loss was {1} caption was {2}"""


if __name__ == "__main__":
    
    vocab, _ = load_glove(vocab_size=100000, embedding_size=300)
    
    with tf.device("/device:GPU:0"):

        with tf.Graph().as_default():

            image_id, image, image_features, object_features, input_seq, target_seq, indicator = (
                import_mscoco(mode="train", batch_size=100, num_epochs=100, is_mini=True))
            image_captioner = ImageCaptioner(300, batch_size=100, beam_size=3, 
                vocab_size=100000, embedding_size=300)
            logits, ids = image_captioner(image_features, object_features, 
                seq_inputs=input_seq, lengths=tf.reduce_sum(indicator, axis=1))
            tf.losses.sparse_softmax_cross_entropy(target_seq, logits, weights=indicator)
            loss = tf.losses.get_total_loss()
            learning_step = tf.train.GradientDescentOptimizer(1.0).minimize(
                loss, var_list=image_captioner.variables)

            captioner_saver = tf.train.Saver(var_list=image_captioner.variables)
            captioner_ckpt = get_image_captioner_checkpoint()

            with tf.Session() as sess:

                if captioner_ckpt is not None:
                    captioner_saver.restore(sess, captioner_ckpt)
                else:
                    sess.run(tf.variables_initializer(
                        image_captioner.variables))

                for i in itertools.count():
                    
                    try:
                        a, b, c, d = sess.run([ids, loss, learning_step, image_features])
                        print(PRINT_STRING.format(i, b, " ".join([
                            vocab.id_to_word(x) for x in a[0, :].tolist()])))
                    except Exception as e:
                        print("Finishing training.")
                        break

                    captioner_saver.save(sess, 'ckpts/caption_model/model.ckpt')
                    