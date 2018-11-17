'''Author: Brandon Trabucco, Copyright 2018
Utilities for manipulating images and captions.'''


import os
import glove
import os.path
import json
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from pycocoapi.coco import COCO
from pycocoapi.eval import COCOEvalCap


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


def load_glove(vocab_size=1000, embedding_size=50):
    
    # The config params for loading the vocab and embedding
    # See: https://github.com/brandontrabucco/glove/tree/8f11a9b3ab927a15a947683ca7a1fcbc5d9c8ba1
    config = glove.configuration.Configuration(
        embedding=embedding_size, filedir="/home/ubuntu/research/data/glove/embeddings/",
        length=vocab_size, start_word="<S>", end_word="</S>", unk_word="<UNK>")
    return glove.load(config)


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


def coco_get_metrics(captions_dict, eval_dir, annotations_file):
    """Get the performance metrics on the dataset.
    """
    time_now = time.time()
    with open(os.path.join(eval_dir, "results." + str(time_now) + ".json"), "w") as f:
        json.dump(captions_dict, f)
    coco = COCO(annotations_file)
    cocoRes = coco.loadRes(os.path.join(eval_dir, "results." + str(time_now) + ".json"))
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    with open(os.path.join(eval_dir, "metrics." + str(time_now) + ".json"), "w") as f:
        metrics_dump = {metric: float(np.sum(score)) for metric, score in cocoEval.eval.items()}
        json.dump(metrics_dump, f)
    return metrics_dump
