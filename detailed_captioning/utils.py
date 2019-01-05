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
import pattern.en


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
    tagger = glove.tagger.load(config)
    return tagger


def cached_load(load_fn):
    """Higher order function for caching.
    """
    
    results = None
    def cached_fn(*args, **kwargs):
        nonlocal results
        if results is None:
            results = load_fn(*args, **kwargs)
        return results
    return cached_fn


@cached_load
def get_visual_categories():
    
    check_runtime()
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
            x = remaining
            x.extend([pluralize(word) for word in remaining])
            fine_grain_words.append(remaining)

    class CategoryMap(object):

        def __init__(self, category_names, category_aliases):

            self.coarse_vocab = {}
            for i, words in enumerate(category_aliases):
                for word in words:
                    self.coarse_vocab[word] = i
                    self.coarse_vocab[pluralize(word)] = i
            self.reverse_coarse_vocab = category_names
            self.fine_vocab = {}
            for words in category_aliases:
                for i, word in enumerate(words):
                    self.fine_vocab[word] = i
                    self.fine_vocab[pluralize(word)] = i
            self.reverse_aliases = category_aliases
            self.plurality_vocab = {}
            for words in category_aliases:
                for word in words:
                    self.plurality_vocab[word] = 0
                    self.plurality_vocab[pluralize(word)] = 1
            self.reverse_plurality_vocab = [[[y, pluralize(y)] for y in x] for x in category_aliases]
            

        def word_to_id(self, word):

            if isinstance(word, list):
                return [self.word_to_id(w) for w in word]
            if (word not in self.coarse_vocab 
                    or word not in self.fine_vocab
                    or word not in self.plurality_vocab):
                return None
            return self.coarse_vocab[word], self.fine_vocab[word], self.plurality_vocab[word]

        def id_to_word(self, course_id, fine_id, plurality):

            if (isinstance(course_id, list) 
                    and isinstance(fine_id, list) 
                    and isinstance(plurality, list)):
                return [self.id_to_word(i, j, k) for i, j, k in zip(course_id, fine_id, plurality)]
            if (not isinstance(course_id, list) 
                    and not isinstance(fine_id, list) 
                    and not isinstance(plurality, list)):
                if (course_id >= 0 and course_id < len(self.reverse_plurality_vocab)
                       and fine_id >= 0 and fine_id < len(self.reverse_plurality_vocab[course_id])
                       and plurality >= 0 and plurality < 2):
                    return self.reverse_plurality_vocab[course_id][fine_id][plurality]
            return None

        def sentence_to_categories(self, sentence):

            if not isinstance(sentence, list):
                return None
            if isinstance(sentence[0], list):
                return [self.sentence_to_categories(x) for x in sentence]
            if not isinstance(sentence[0], str):
                return None
            coarse = []
            fine = []
            plurality = []
            for i, x in enumerate(self.reverse_plurality_vocab):
                for j, y in enumerate(x):
                    for k, word in enumerate(y):
                        if word in sentence:
                            coarse.append(i)
                            fine.append(j)
                            plurality.append(k)
            return coarse, fine, plurality
            
    return CategoryMap(word_names, fine_grain_words)


@cached_load
def get_visual_attributes():
    
    check_runtime()
    filename = "data/visual_words.txt"
    attribute_names = []
    with open(filename, "r") as f:
        content = f.readlines()
    for line in content:
        if line is not None:
            line = line.lower().strip()
        if line is not None:
            first, *remaining = line.split(", ")
            if first not in attribute_names:
                attribute_names.append(first)
            plural_first = pluralize(first)
            if plural_first not in attribute_names:
                attribute_names.append(plural_first)
            for rest in remaining:
                if rest not in attribute_names:
                    attribute_names.append(rest)
                plural_rest = pluralize(rest)
                if plural_rest not in attribute_names:
                    attribute_names.append(plural_rest)
                    
    class AttributeMap(object):

        def __init__(self, attribute_names):
            
            self.vocab = { word : i for i, word in enumerate(attribute_names)}
            self.reverse_vocab = attribute_names

        def word_to_id(self, word):

            if isinstance(word, list):
                return [self.word_to_id(w) for w in word]
            if word not in self.vocab:
                return None
            return self.vocab[word]

        def id_to_word(self, index):

            if isinstance(index, list):
                return [self.id_to_word(i) for i in index]
            if index < 0 or index >= len(self.reverse_vocab):
                return None
            return self.reverse_vocab[index]

        def sentence_to_attributes(self, sentence):

            if not isinstance(sentence, list):
                return None
            if isinstance(sentence[0], list):
                return [self.sentence_to_categories(x) for x in sentence]
            present_attributes = []
            for i, word in enumerate(self.reverse_vocab):
                if word in sentence:
                    present_attributes.append(i)
            return present_attributes
            
    return AttributeMap(attribute_names)


@cached_load
def get_parts_of_speech():
    
    check_runtime
    POS_names = ["NOUN", "VERB", "ADJ", 
                       "NUM", "ADV", "PRON", 
                       "PRT", "ADP", "DET", 
                       "CONJ", ".", "X", "<S>", "</S>", "<UNK>"]
                    
    class PartOfSpeechMap(object):

        def __init__(self, POS_names):
            
            self.vocab = { word : i for i, word in enumerate(POS_names)}
            self.reverse_vocab = POS_names
            self.start_id = self.vocab["<S>"]
            self.end_id   = self.vocab["</S>"]
            self.unk_id   = self.vocab["<UNK>"]

        def word_to_id(self, word, tagger):

            if isinstance(word, list):
                return [self.word_to_id(w, tagger) for w in word]
            if word in self.vocab:
                self.vocab[word]
            _, this_POS = tagger.tag([word])[0]
            if this_POS in self.vocab:
                return self.vocab[this_POS]
            return None

        def id_to_word(self, index):

            if isinstance(index, list):
                return [self.id_to_word(i) for i in index]
            if index < 0 or index >= len(self.reverse_vocab):
                return None
            return self.reverse_vocab[index]
            
    return PartOfSpeechMap(POS_names)


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


def get_up_down_attribute_checkpoint():

    check_runtime()
    name = 'ckpts/up_down_attribute/'
    tf.gfile.MakeDirs(name)
    return tf.train.latest_checkpoint(name), (name + 'model.ckpt')


def get_visual_sentinel_checkpoint():

    check_runtime()
    name = 'ckpts/visual_sentinel/'
    tf.gfile.MakeDirs(name)
    return tf.train.latest_checkpoint(name), (name + 'model.ckpt')


def get_visual_sentinel_attribute_checkpoint():

    check_runtime()
    name = 'ckpts/visual_sentinel_attribute/'
    tf.gfile.MakeDirs(name)
    return tf.train.latest_checkpoint(name), (name + 'model.ckpt')


def get_show_and_tell_checkpoint():

    check_runtime()
    name = 'ckpts/show_and_tell/'
    tf.gfile.MakeDirs(name)
    return tf.train.latest_checkpoint(name), (name + 'model.ckpt')


def get_show_and_tell_attribute_checkpoint():

    check_runtime()
    name = 'ckpts/show_and_tell_attribute/'
    tf.gfile.MakeDirs(name)
    return tf.train.latest_checkpoint(name), (name + 'model.ckpt')


def get_show_attend_and_tell_attribute_checkpoint():

    check_runtime()
    name = 'ckpts/show_attend_and_tell_attribute/'
    tf.gfile.MakeDirs(name)
    return tf.train.latest_checkpoint(name), (name + 'model.ckpt')


def get_show_attend_and_tell_checkpoint():

    check_runtime()
    name = 'ckpts/show_attend_and_tell/'
    tf.gfile.MakeDirs(name)
    return tf.train.latest_checkpoint(name), (name + 'model.ckpt')


def get_show_attend_and_tell_attribute_checkpoint():

    check_runtime()
    name = 'ckpts/show_attend_and_tell_attribute/'
    tf.gfile.MakeDirs(name)
    return tf.train.latest_checkpoint(name), (name + 'model.ckpt')


def get_spatial_attention_checkpoint():

    check_runtime()
    name = 'ckpts/spatial_attention/'
    tf.gfile.MakeDirs(name)
    return tf.train.latest_checkpoint(name), (name + 'model.ckpt')


def get_spatial_attention_attribute_checkpoint():

    check_runtime()
    name = 'ckpts/spatial_attention_attribute/'
    tf.gfile.MakeDirs(name)
    return tf.train.latest_checkpoint(name), (name + 'model.ckpt')


def get_best_first_checkpoint():

    check_runtime()
    name = 'ckpts/best_first/'
    tf.gfile.MakeDirs(name)
    return tf.train.latest_checkpoint(name), (name + 'model.ckpt')


def get_attribute_detector_checkpoint():

    check_runtime()
    name = 'ckpts/attribute_detector/'
    tf.gfile.MakeDirs(name)
    return tf.train.latest_checkpoint(name), (name + 'model.ckpt')


def get_up_down_part_of_speech_checkpoint():
    
    check_runtime()
    name = 'ckpts/up_down_part_of_speech/'
    tf.gfile.MakeDirs(name)
    return tf.train.latest_checkpoint(name), (name + 'model.ckpt')


def remap_decoder_name_scope(var_list):
    """Bug fix for running beam search decoder and dynamic rnn during inference / training."""
    var_names = {}
    for x in var_list:
        x_name = x.name
        if "decoder" in x_name and "logits" in x_name:
            var_names[x_name.replace("decoder/", "")
                .replace("decoder_1/", "")
                .replace("decoder_2/", "")
                .replace("decoder_3/", "")
                .replace("decoder_4/", "")[:-2] ] = x
        else:
            var_names[x_name.replace("decoder/", "rnn/")
                .replace("decoder_1/", "rnn/")
                .replace("decoder_2/", "rnn/")
                .replace("decoder_3/", "rnn/")
                .replace("decoder_4/", "rnn/")[:-2] ] = x
    return var_names


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


def pluralize(word):
    """
    Converts a word to its plural form.
    Adapted from https://stackoverflow.com/questions/18902608/generating-the-plural-form-of-a-noun
    """
    return pattern.en.pluralize(word)
