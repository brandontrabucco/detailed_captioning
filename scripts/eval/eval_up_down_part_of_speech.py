'''Author: Brandon Trabucco, Copyright 2019
Test the image captioning model with some fake inputs.'''


import time
import itertools
import tensorflow as tf
import numpy as np
from detailed_captioning.utils import get_parts_of_speech
from detailed_captioning.layers.part_of_speech_image_captioner import PartOfSpeechImageCaptioner 
from detailed_captioning.cells.up_down_cell import UpDownCell
from detailed_captioning.utils import check_runtime
from detailed_captioning.utils import load_glove
from detailed_captioning.utils import get_up_down_part_of_speech_checkpoint
from detailed_captioning.utils import remap_decoder_name_scope
from detailed_captioning.utils import list_of_ids_to_string
from detailed_captioning.utils import recursive_ids_to_string
from coco_metrics import evaluate
from detailed_captioning.utils import get_train_annotations_file
from detailed_captioning.utils import get_val_annotations_file
from detailed_captioning.inputs.mean_image_and_object_features_and_parts_of_speech_only import import_mscoco


PRINT_STRING = """({3:.2f} img/sec) iteration: {0:05d}\n    caption: {1}\n    label: {2}"""
tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_integer("batch_size", 1, "")
tf.flags.DEFINE_integer("beam_size", 3, "")
tf.flags.DEFINE_boolean("is_mini", False, "")
tf.flags.DEFINE_string("mode", "eval", "")
FLAGS = tf.flags.FLAGS


if __name__ == "__main__":
    
    vocab, pretrained_matrix = load_glove(vocab_size=100000, embedding_size=300)
    pos, pos_embeddings = get_parts_of_speech(), np.random.normal(0, 0.1, [15, 300])
    with tf.Graph().as_default():

        (image_id, image_features, object_features, input_seq, target_seq, indicator, 
                pos_input_seq, pos_target_seq, pos_indicator) = import_mscoco(
            mode=FLAGS.mode, batch_size=FLAGS.batch_size, num_epochs=1, is_mini=FLAGS.is_mini)
        up_down_caption_cell = UpDownCell(300, name="up_down_caption_cell")
        up_down_decoder_cell = UpDownCell(300, name="up_down_decoder_cell")
        up_down_encoder_cell = UpDownCell(300, name="up_down_encoder_cell")
        image_captioner = PartOfSpeechImageCaptioner(
            up_down_caption_cell, vocab, pretrained_matrix,
            up_down_decoder_cell, up_down_encoder_cell, pos, pos_embeddings)
        pos_logits, pos_ids, word_logits, word_ids = image_captioner(
            mean_image_features=image_features, 
            mean_object_features=object_features)
        
        captioner_saver = tf.train.Saver(var_list=remap_decoder_name_scope(image_captioner.variables))
        captioner_ckpt, captioner_ckpt_name = get_up_down_part_of_speech_checkpoint()
        

        with tf.Session() as sess:

            assert(captioner_ckpt is not None)
            captioner_saver.restore(sess, captioner_ckpt)
            used_ids = set()
            json_dump = []

            for i in itertools.count():
                time_start = time.time()
                try:
                    _ids, _target_seq, _image_id = sess.run([word_ids, target_seq, image_id])
                except:
                    break
                the_captions = recursive_ids_to_string(_ids[:, 0, :].tolist(), vocab)
                the_labels = recursive_ids_to_string(_target_seq[:, :].tolist(), vocab)
                the_image_ids = _image_id.tolist()
                for j, x, y in zip(the_image_ids, the_captions, the_labels):
                    if not j in used_ids:
                        used_ids.add(j)
                        json_dump.append({"image_id": j, "caption": x})
                print(PRINT_STRING.format(i, the_captions[0], the_labels[0], 
                    FLAGS.batch_size / (time.time() - time_start))) 

            print("Finishing evaluating.")
            evaluate(
                FLAGS.mode,
                json_dump, 
                captioner_ckpt_name.replace("model.ckpt", ""), 
                (get_train_annotations_file() if FLAGS.mode in ["train", "eval"] 
                    else get_val_annotations_file()))
