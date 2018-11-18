'''Author: Brandon Trabucco, Copyright 2019
Test the image captioning model with some fake inputs.'''


import time
import itertools
import tensorflow as tf
import numpy as np
from detailed_captioning.layers.image_captioner import ImageCaptioner
from detailed_captioning.cells.up_down_cell import UpDownCell
from detailed_captioning.utils import check_runtime
from detailed_captioning.utils import load_glove
from detailed_captioning.utils import get_up_down_checkpoint
from detailed_captioning.utils import remap_decoder_name_scope
from detailed_captioning.utils import list_of_ids_to_string
from detailed_captioning.utils import recursive_ids_to_string
from detailed_captioning.utils import coco_get_metrics
from detailed_captioning.utils import get_train_annotations_file
from detailed_captioning.inputs.mean_image_and_object_features_only import import_mscoco


if __name__ == "__main__":
    
    vocab, pretrained_matrix = load_glove(vocab_size=100000, embedding_size=300)
    with tf.Graph().as_default():

        image_id, mean_features, object_features, input_seq, target_seq, indicator = (
            import_mscoco(mode="train", batch_size=10, num_epochs=1, is_mini=True))
        image_captioner = ImageCaptioner(UpDownCell(300), vocab, pretrained_matrix, 
            trainable=False, beam_size=16)
        logits, ids = image_captioner(mean_image_features=mean_features,
            mean_object_features=object_features)
        captioner_saver = tf.train.Saver(var_list=remap_decoder_name_scope(image_captioner.variables))
        captioner_ckpt, captioner_ckpt_name = get_up_down_checkpoint()

        with tf.Session() as sess:

            assert(captioner_ckpt is not None)
            captioner_saver.restore(sess, captioner_ckpt)
            used_ids = set()
            json_dump = []

            for i in itertools.count():
                try:
                    _ids, _target_seq, _image_id = sess.run([ids, target_seq, image_id])
                except:
                    break
                the_captions = recursive_ids_to_string(_ids[:, 0, :].tolist(), vocab)
                the_labels = recursive_ids_to_string(_target_seq[:, :].tolist(), vocab)
                the_image_ids = _image_id.tolist()
                for j, x, y in zip(the_image_ids, the_captions, the_labels):
                    if not j in used_ids:
                        used_ids.add(j)
                        json_dump.append({"image_id": j, "caption": x})
                print("Iteration {0}".format(i)) 

            print("Finishing evaluating.")
            coco_get_metrics(json_dump, "ckpts/up_down/", get_train_annotations_file())
