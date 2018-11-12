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
from detailed_captioning.utils import remap_decoder_name_scope
from detailed_captioning.utils import list_of_ids_to_string
from detailed_captioning.utils import recursive_ids_to_string
from detailed_captioning.inputs.mscoco import import_mscoco


BATCH_SIZE = 4
            

if __name__ == "__main__":
    
    vocab, _ = load_glove(vocab_size=100000, embedding_size=300)
    
    with tf.device("/cpu:0"):

        with tf.Graph().as_default():

            image_id, image, scores, boxes, input_seq, target_seq, indicator = (
                import_mscoco(is_training=False, batch_size=BATCH_SIZE, num_epochs=1, k=8))
            
            image_captioner = ImageCaptioner(300, trainable=False, batch_size=BATCH_SIZE, beam_size=3, 
                vocab_size=100000, embedding_size=300, fine_tune_cnn=False)
            logits, ids = image_captioner(image, boxes)

            captioner_saver = tf.train.Saver(var_list=remap_decoder_name_scope(
                image_captioner.variables.all))
            captioner_ckpt = get_image_captioner_checkpoint()

            with tf.Session() as sess:

                assert(captioner_ckpt is not None)
                captioner_saver.restore(sess, captioner_ckpt)
                a, b, c = sess.run([ids, target_seq, image_id])

                for i in itertools.count():
                    
                    try:
                        a, b, c = sess.run([ids, target_seq, image_id])
                    except Exception as e:
                        print("Finishing evaluating.".format(e))
                        break
                        
                    # make sure to convert numpy to lists
                    the_captions = recursive_ids_to_string(a[:, 0, :].tolist(), vocab)
                    the_labels = recursive_ids_to_string(b[:, :].tolist(), vocab)
                    the_image_ids = c.tolist()
                    for j, x, y in zip(the_image_ids, the_captions, the_labels):
                        print("Image {0} caption was: {1} ; label was: {2}".format(
                            j, x, y)) 
