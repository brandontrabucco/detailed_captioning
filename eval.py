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


BATCH_SIZE = 4


def _remap_decoder_name_scope(var_list):
    """Bug fix for running beam search decoder and dynamic rnn."""
    return {x.name.replace("decoder/", "rnn/")[:-2] : x for x in var_list}


def list_of_ids_to_string(ids, vocab):
    """Converts the list of ids to a string of captions.
    Args:
        ids - a flat list of word ids.
        vocab - a glove vocab object.
    Returns:
        string of words in vocabulary.
    """
    assert(isinstance(ids, list))
    result = ""
    for i in ids:
        if i == vocab.end_id:
            return result
        result = result + " " + str(vocab.id_to_word(i))
        if i == vocab.word_to_id("."):
            return result
    return result.strip().lower()


def recursive_ids_to_string(ids, vocab):
    """Converts the list of ids to a string of captions.
    Args:
        ids - a nested list of word ids.
        vocab - a glove vocab object.
    Returns:
        string of words in vocabulary.
    """
    assert(isinstance(ids, list))
    if isinstance(ids[0], list):
        return [recursive_ids_to_string(x, vocab) for x in ids]
    return list_of_ids_to_string(ids, vocab)
            

if __name__ == "__main__":
    
    vocab, _ = load_glove(vocab_size=100000, embedding_size=300)
    
    with tf.device("/cpu:0"):

        with tf.Graph().as_default():

            image_id, image, scores, boxes, input_seq, target_seq, indicator = (
                import_mscoco(is_training=False, batch_size=BATCH_SIZE, num_epochs=1, k=8))
            
            image_captioner = ImageCaptioner(300, trainable=False, batch_size=BATCH_SIZE, beam_size=3, 
                vocab_size=100000, embedding_size=300, fine_tune_cnn=False)
            logits, ids = image_captioner(image, boxes, seq_inputs=input_seq, 
                                          lengths=tf.reduce_sum(indicator, axis=1))

            tf.losses.sparse_softmax_cross_entropy(target_seq, logits, weights=indicator)
            loss = tf.losses.get_total_loss()

            captioner_saver = tf.train.Saver(
                var_list=_remap_decoder_name_scope(image_captioner.variables.all))
            captioner_ckpt = get_image_captioner_checkpoint()

            with tf.Session() as sess:

                assert(captioner_ckpt is not None)
                captioner_saver.restore(sess, captioner_ckpt)

                for i in itertools.count():
                    
                    try:
                        a, b, c = sess.run([ids, target_seq, image_id])
                        the_captions = recursive_ids_to_string(a, vocab)
                        the_labels = recursive_ids_to_string(b, vocab)
                        for x, y in zip(the_captions, the_labels):
                            print("Caption was: {0}  |  Label was: {1}".format(
                                x[0], y)) # inference runs beam search on x (extra dim)
                        
                    except Exception as e:
                        print("Finishing evaluating e = {0}.".format(str(e)))
                        break

                    