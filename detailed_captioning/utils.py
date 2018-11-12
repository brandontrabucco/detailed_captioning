'''Author: Brandon Trabucco, Copyright 2018
Utilities for manipulating images and captions.'''


import os
import glove
import os.path
import numpy as np
from PIL import Image
import tensorflow as tf
from object_detection.utils import label_map_util


def check_runtime():
    
    is_okay = False
    if os.path.isfile('the_magic_number_is'):
        with open('the_magic_number_is', 'r') as f:
            is_okay = int(f.read()) == 314159
    if not is_okay:
        raise Exception('Run this script from the library root directory.')


def load_image_from_path(image_path):
    
    image = Image.open(image_path)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def load_glove(vocab_size=1000, embedding_size=50):
    
    # The config params for loading the vocab and embedding
    # See: https://github.com/brandontrabucco/glove/tree/8f11a9b3ab927a15a947683ca7a1fcbc5d9c8ba1
    config = glove.configuration.Configuration(
        embedding=embedding_size, filedir="/home/ubuntu/research/data/glove/embeddings/",
        length=vocab_size, start_word="<S>", end_word="</S>", unk_word="<UNK>")
    return glove.load(config)


def get_faster_rcnn_config():
    
    check_runtime()
    return ('ckpts/' + 
             'faster_rcnn_resnet101_coco/' +
             'pipeline.config')


def get_faster_rcnn_checkpoint():

    check_runtime()
    return tf.train.latest_checkpoint('ckpts/' + 
             'faster_rcnn_resnet101_coco/')


def get_resnet_v2_101_checkpoint():

    check_runtime()
    return ('ckpts/resnet_v2_101/resnet_v2_101.ckpt')


def get_image_captioner_checkpoint():

    check_runtime()
    return tf.train.latest_checkpoint('ckpts/' + 
             'caption_model/')


def remap_decoder_name_scope(var_list):
    """Bug fix for running beam search decoder and dynamic rnn."""
    return {
        x.name.replace("decoder/", "rnn/")
              .replace("rnn/logits_layer", "logits_layer")
              .replace("decoder/while/BeamSearchDecoderStep/logits_layer", "logits_layer")[:-2] : x 
        for x in var_list}


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
