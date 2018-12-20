'''Author: Brandon Trabucco, Copyright 2018
Utilities for manipulating images and captions.'''


import os
import glove
import glove.configuration
import glove.heuristic
import glove.tagger
import os.path
import json
import time
import numpy as np
import tensorflow as tf
import pickle as pkl
import nltk
from PIL import Image
from nltk.corpus import brown


class CategoricalMap(object):

    def __init__(self, category_names, category_aliases):

        if isinstance(category_aliases[0], list):
            self.vocab = {}
            for i, words in enumerate(category_aliases):
                for word in words:
                    self.vocab = { word : i }
        if isinstance(category_aliases[0], str):
            self.vocab = { word : i for i, word in enumerate(category_aliases)}
        self.reverse_vocab = category_names

    def sentence_to_categories(self, list_of_words):
        
        if len(list_of_words) > 0:
            if isinstance(list_of_words[0], list):
                return [self.sentence_to_categories(x) for x in list_of_words]
            if not isinstance(list_of_words[0], str):
                return None
            return [1.0 if x in list_of_words else 0.0 for x in self.reverse_vocab]


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


def tile_with_new_axis(tensor, repeats, locations):
    
    nd = zip(repeats, locations)
    nd = sorted(nd, key=lambda ab: ab[1])
    repeats, locations = zip(*nd)
    for i in sorted(locations):
        tensor = tf.expand_dims(tensor, i)
    reverse_d = {val: idx for idx, val in enumerate(locations)}
    tiles = [repeats[reverse_d[i]] if i in locations else 1 for i, _s in enumerate(tensor.shape)]
    return tf.tile(tensor, tiles)


def collapse_dims(tensor, flat_points):
    
    flat_size = tf.shape(tensor)[flat_points[0]]
    for i in flat_points[1:]:
        flat_size = flat_size * tf.shape(tensor)[i]
    fixed_points = [i for i in range(len(tensor.shape)) if i not in flat_points]
    fixed_shape = [tf.shape(tensor)[i] for i in fixed_points]
    tensor = tf.transpose(tensor, fixed_points + flat_points)
    final_points = list(range(len(fixed_shape)))
    final_points.insert(flat_points[0], len(fixed_shape))
    return tf.transpose(tf.reshape(tensor, fixed_shape + [flat_size]), final_points)


def load_glove(vocab_size=100000, embedding_size=300):
    
    # The config params for loading the vocab and embedding
    # See: https://github.com/brandontrabucco/glove/tree/8f11a9b3ab927a15a947683ca7a1fcbc5d9c8ba1
    config = glove.configuration.Configuration(
        embedding=embedding_size, filedir="/home/ubuntu/research/data/glove/embeddings/",
        length=vocab_size, start_word="<S>", end_word="</S>", unk_word="<UNK>")
    return glove.load(config)


def load_tagger():
    
    # The config params for loading the POS tagger
    # See: https://github.com/brandontrabucco/glove/tree/8f11a9b3ab927a15a947683ca7a1fcbc5d9c8ba1
    config = glove.configuration.TaggerConfiguration(
        tagger_dir="/home/ubuntu/research/data/glove/tagger/")
    return glove.tagger.load(config)


def get_visual_words():
    
    filename = "data/visual_words.txt"
    word_names = []
    fine_grain_words = []
    with open(filename, "r") as f:
        content = f.readlines()
    for line in content:
        if line is not None:
            line = line.lower().strip()
        if line is not None:
            first, *remaining = line.split(", ")
            word_names.append(first)
            fine_grain_words.append(remaining)
            
    return CategoricalMap(word_names, fine_grain_words)


def get_faster_rcnn_config():
    
    check_runtime()
    return ('ckpts/faster_rcnn_resnet101_coco/pipeline.config')


def get_faster_rcnn_checkpoint():

    check_runtime()
    return tf.train.latest_checkpoint('ckpts/faster_rcnn_resnet101_coco/')


def get_resnet_v2_101_checkpoint():

    check_runtime()
    return ('ckpts/resnet_v2_101/resnet_v2_101.ckpt')


def get_up_down_checkpoint():

    check_runtime()
    name = 'ckpts/up_down/'
    tf.gfile.MakeDirs(name)
    return tf.train.latest_checkpoint(name), (name + 'model.ckpt')


def get_visual_sentinel_checkpoint():

    check_runtime()
    name = 'ckpts/visual_sentinel/'
    tf.gfile.MakeDirs(name)
    return tf.train.latest_checkpoint(name), (name + 'model.ckpt')


def get_show_and_tell_checkpoint():

    check_runtime()
    name = 'ckpts/show_and_tell/'
    tf.gfile.MakeDirs(name)
    return tf.train.latest_checkpoint(name), (name + 'model.ckpt')


def get_show_attend_and_tell_checkpoint():

    check_runtime()
    name = 'ckpts/show_attend_and_tell/'
    tf.gfile.MakeDirs(name)
    return tf.train.latest_checkpoint(name), (name + 'model.ckpt')


def get_spatial_attention_checkpoint():

    check_runtime()
    name = 'ckpts/spatial_attention/'
    tf.gfile.MakeDirs(name)
    return tf.train.latest_checkpoint(name), (name + 'model.ckpt')


def get_best_first_checkpoint():

    check_runtime()
    name = 'ckpts/best_first/'
    tf.gfile.MakeDirs(name)
    return tf.train.latest_checkpoint(name), (name + 'model.ckpt')


def remap_decoder_name_scope(var_list):
    """Bug fix for running beam search decoder and dynamic rnn."""
    return {
        x.name.replace("decoder/", "rnn/")
              .replace("rnn/logits_layer", "logits_layer")
              .replace("decoder/while/BeamSearchDecoderStep/logits_layer", "logits_layer")[:-2] : x 
        for x in var_list}


def get_train_annotations_file():
    
    check_runtime()
    return ('data/coco/annotations/captions_train2017.json')


def get_val_annotations_file():
    
    check_runtime()
    return ('data/coco/annotations/captions_val2017.json')


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
