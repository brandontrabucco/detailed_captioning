'''Author: Brandon Trabucco, Copyright 2019
Test the image captioning model with some fake inputs.'''


import os
import time
import json
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
from detailed_captioning.utils import coco_get_metrics
from detailed_captioning.utils import get_train_annotations_file
from detailed_captioning.inputs.mscoco import import_mscoco


if __name__ == "__main__":
    
    vocab, _ = load_glove(vocab_size=100000, embedding_size=300)
    
    with tf.device("/cpu:0"):

        with tf.Graph().as_default():

            image_id, image, image_features, object_features, input_seq, target_seq, indicator = (
                import_mscoco(mode="train", batch_size=50, num_epochs=1, is_mini=True))
            image_captioner = ImageCaptioner(300, trainable=False, batch_size=50, beam_size=3, 
                vocab_size=100000, embedding_size=300)
            logits, ids = image_captioner(image_features, object_features)
            captioner_saver = tf.train.Saver(var_list=remap_decoder_name_scope(image_captioner.variables))
            captioner_ckpt = get_image_captioner_checkpoint()

            with tf.Session() as sess:

                assert(captioner_ckpt is not None)
                captioner_saver.restore(sess, captioner_ckpt)
                used_ids = set()
                json_dump = []

                for i in itertools.count():
                    
                    try:
                        a, b, c = sess.run([ids, target_seq, image_id])
                    except Exception as e:
                        print("Finishing evaluating.".format(e))
                        break
                    the_captions = recursive_ids_to_string(a[:, 0, :].tolist(), vocab)
                    the_labels = recursive_ids_to_string(b[:, :].tolist(), vocab)
                    the_image_ids = c.tolist()
                    for j, x, y in zip(the_image_ids, the_captions, the_labels):
                        if not j in used_ids:
                            used_ids.add(j)
                            json_dump.append({"image_id": j, "caption": x})
                    print("Iteration {0}".format(i)) 
                        
                coco_get_metrics(json_dump, "./", get_train_annotations_file())
