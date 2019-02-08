'''Author: Brandon Trabucco, Copyright 2019
Gets the most frequent words in the MSCOCO dataset.'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import os.path
import random
import sys
import threading
import nltk.tokenize
import tensorflow as tf
from datetime import datetime
from detailed_captioning.utils import load_tagger


tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_string("train_captions_file", "/tmp/captions_train2014.json",
                       "Training captions JSON file.")
tf.flags.DEFINE_string("val_captions_file", "/tmp/captions_val2014.json",
                       "Validation captions JSON file.")
tf.flags.DEFINE_string("output_dir", "/tmp/", "Output data directory.")
tf.flags.DEFINE_string("filename", "word_names.txt", "Name of the resulting file.")
tf.flags.DEFINE_string("part_of_speech", "None", "")
tf.flags.DEFINE_integer("top_k_words", 1000, "")
FLAGS = tf.flags.FLAGS


def _load_and_process_metadata(captions_file, tagger, k=1000, part_of_speech=None):
    """Loads the image annotations and returnsd a list of the most k frequent parts of speech.
    Args:
        captions_file: str, path to the caption annotations file.
        k: int, the number of words to select.
        part_of_speech: str or None, the key to use to filter words.
    Returns: a list of words sorted by frequency.
    """
    with tf.gfile.FastGFile(captions_file, "rb") as f:
        caption_data = json.load(f)

    # Extract the captions. Each image_id is associated with multiple captions.
    word_frequencies = {}
    for annotation in caption_data["annotations"]:
        tokens = nltk.tokenize.word_tokenize(annotation["caption"].lower().strip())
        parts_of_speech = tagger.tag(tokens)
        for word, pos in parts_of_speech:
            if part_of_speech is None or part_of_speech == "None" or pos == part_of_speech:
                word_frequencies.setdefault(word, 0)
                word_frequencies[word] += 1
                
    sorted_words = list(zip(*list(sorted(
        word_frequencies.items(), reverse=True, key=lambda x: x[1]))))[0]
    
    return sorted_words[:k]


def main(unused_argv):
        
    tagger = load_tagger()
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    # Load image metadata from caption files.
    list_of_names1 = _load_and_process_metadata(FLAGS.train_captions_file, tagger, 
        k=FLAGS.top_k_words, part_of_speech=FLAGS.part_of_speech)
    list_of_names2 = _load_and_process_metadata(FLAGS.val_captions_file, tagger, 
        k=FLAGS.top_k_words, part_of_speech=FLAGS.part_of_speech)
    
    with open(os.path.join(FLAGS.output_dir, FLAGS.filename), "w") as f:
        f.write("\n".join(list_of_names1))


if __name__ == "__main__":
    tf.app.run()