'''Author: Brandon Trabucco, Copyright 2019
Test the image captioning model with some fake inputs.'''


import tensorflow as tf
import numpy as np
from detailed_captioning.layers.image_captioner import ImageCaptioner
from detailed_captioning.utils import check_runtime
from detailed_captioning.utils import load_glove
from detailed_captioning.utils import load_image_from_path
from detailed_captioning.utils import get_object_detector_config
from detailed_captioning.utils import get_object_detector_checkpoint
from detailed_captioning.utils import get_image_captioner_checkpoint


if __name__ == "__main__":

    check_runtime()
    vocab, pretrained_matrix = load_glove(vocab_size=100, embedding_size=50)
    
    images = np.random.normal(0, 1, [1, 640, 640, 3])
    test_string = ["<S>", "thanks", "for", "reading", "my", "code", ".", "</S>"]
    test_ids = [vocab.word_to_id(x) for x in test_string]
    
    with tf.Graph().as_default():
        
        image_tensor = tf.placeholder(
            tf.float32, name="image_tensor", shape=[1, 640, 640, 3])
        input_tensor = tf.placeholder(
            tf.int32, name='input_tensor', shape=[1, None])
        label_tensor = tf.placeholder(
            tf.int32, name='label_tensor', shape=[1, None])
        length_tensor = tf.placeholder(
            tf.int32, name='label_tensor', shape=[1])
        
        captioner = ImageCaptioner(
            get_object_detector_config(), 50,
            batch_size=1, beam_size=3, vocab_size=100, embedding_size=50)
        
        logits, ids = captioner(image_tensor, seq_inputs=input_tensor, 
                                lengths=length_tensor - 1)
    
        with tf.Session() as sess:

            tf.losses.sparse_softmax_cross_entropy(label_tensor, logits)
            loss = tf.losses.get_total_loss()
            learning_step = tf.train.AdamOptimizer().minimize(
                loss, var_list=captioner.variables.captioner_variables)

            # Load the model from a checkpoint
            captioner_ckpt = get_image_captioner_checkpoint()
            all_saver = tf.train.Saver(var_list=captioner.variables.join())
            if captioner_ckpt is not None:
                all_saver.restore(sess, captioner_ckpt)
            else:
                saver = tf.train.Saver(var_list=captioner.variables.detector_variables)
                saver.restore(sess, get_object_detector_checkpoint())
                sess.run(tf.variables_initializer(
                    list(captioner.variables.captioner_variables.values())))
            
            l = sess.run(ids, feed_dict={ 
                image_tensor: images, 
                input_tensor: [test_ids[:-1]], 
                label_tensor: [test_ids[1:]], 
                length_tensor: [len(test_ids) - 1] })
            
            print(l)
            
            all_saver.save(sess, 'ckpts/caption_model/model.ckpt')
            