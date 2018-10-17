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


def get_mscoco_label_map():
    
    check_runtime()
    map_file = os.path.join('data/', 'mscoco_label_map.pbtxt')
    return label_map_util.create_category_index_from_labelmap(
        map_file, use_display_name=True)


def load_glove(vocab_size=1000, embedding_size=50):
    
    # The config params for loading the vocab and embedding
    # See: https://github.com/brandontrabucco/glove/tree/8f11a9b3ab927a15a947683ca7a1fcbc5d9c8ba1
    config = glove.configuration.Configuration(
        embedding=embedding_size, filedir="/home/ubuntu/research/data/glove/embeddings/",
        length=vocab_size, start_word="<S>", end_word="</S>", unk_word="<UNK>")
    return glove.load(config)


def get_object_detector_config():
    
    check_runtime()
    return ('ckpts/' + 
             'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync/' +
             'pipeline.config')


def get_object_detector_checkpoint():

    check_runtime()
    return tf.train.latest_checkpoint('ckpts/' + 
             'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync/')


def get_image_captioner_checkpoint():

    check_runtime()
    return tf.train.latest_checkpoint('ckpts/' + 
             'caption_model/')
