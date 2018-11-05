'''Author: Brandon Trabucco, Copyright 2019
Builds TF Records for the MSCOCO 2017 dataset'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import os.path
import random
import sys
import threading


import nltk.tokenize
import numpy as np
import tensorflow as tf
from six.moves import xrange
from collections import Counter
from collections import namedtuple
from datetime import datetime
from detailed_captioning.utils import load_glove
from detailed_captioning.layers.box_extractor import BoxExtractor
from detailed_captioning.utils import get_faster_rcnn_config
from detailed_captioning.utils import get_faster_rcnn_checkpoint


tf.flags.DEFINE_string("train_image_dir", "/tmp/train2014/",
                       "Training image directory.")
tf.flags.DEFINE_string("val_image_dir", "/tmp/val2014",
                       "Validation image directory.")

tf.flags.DEFINE_string("train_captions_file", "/tmp/captions_train2014.json",
                       "Training captions JSON file.")
tf.flags.DEFINE_string("val_captions_file", "/tmp/captions_val2014.json",
                       "Validation captions JSON file.")

tf.flags.DEFINE_string("output_dir", "/tmp/", "Output data directory.")

tf.flags.DEFINE_integer("train_shards", 256,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 4,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 8,
                        "Number of shards in testing TFRecord files.")

tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the images.")

tf.flags.DEFINE_string("start_word", "<S>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</S>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")

tf.flags.DEFINE_integer("vocab_size", 100000,
                       "Special word meaning 'unknown'.")
tf.flags.DEFINE_integer("embedding_size", 300,
                       "Special word meaning 'unknown'.")

FLAGS = tf.flags.FLAGS

ImageMetadata = namedtuple("ImageMetadata",
                           ["image_id", "filename", "captions"])

PreextractedMetadata = namedtuple("PreextractedMetadata",
                           ["image_id", "filename", "captions", "scores", "boxes"])

class ImageDecoder(object):
    """Helper class for decoding images in TensorFlow."""

    def __init__(self):
        # Create a single TensorFlow Session for all image decoding calls.
        self._sess = tf.Session()

        # TensorFlow ops for JPEG decoding.
        self._encoded_jpeg = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)
        self._decode_jpeg = tf.image.resize_images(self._decode_jpeg, [640, 640])

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _float_feature(value):
    """Wrapper for inserting an float Feature into a SequenceExample proto."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature_list(values):
    """Wrapper for inserting an float64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_float_feature(v) for v in values])


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _to_sequence_example(image, vocab):
    """Builds a SequenceExample proto for an image-caption pair.
    Args:
        image: An ImageMetadata object.
        vocab: A Vocabulary object.
    Returns:
        A SequenceExample proto.
    """
    with tf.gfile.FastGFile(image.filename, "rb") as f:
        encoded_image = f.read()

    context = tf.train.Features(feature={
        "image/image_id": _int64_feature(image.image_id),
        "image/data": _bytes_feature(encoded_image),
    })
    assert len(image.captions) == 1
    caption = image.captions[0]
    caption_ids = [vocab.word_to_id(word) for word in caption]
    feature_lists = tf.train.FeatureLists(feature_list={
        "image/caption": _bytes_feature_list([bytes(c, "utf-8") for c in caption]),
        "image/caption_ids": _int64_feature_list(caption_ids),
        "image/scores": _float_feature_list(image.scores.flatten().tolist()),
        "image/scores_shape": _int64_feature_list(image.scores.shape),
        "image/boxes": _float_feature_list(image.boxes.flatten().tolist()),
        "image/boxes_shape": _int64_feature_list(image.boxes.shape),
    })
    sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)

    return sequence_example


def _process_image_files(thread_index, ranges, name, images, vocab, num_shards):
    """Processes and saves a subset of images as TFRecord files in one thread.
    Args:
      thread_index: Integer thread identifier within [0, len(ranges)].
      ranges: A list of pairs of integers specifying the ranges of the dataset to
        process in parallel.
      name: Unique identifier specifying the dataset.
      images: List of ImageMetadata.
      vocab: A Vocabulary object.
      num_shards: Integer number of shards for the output files.
    """
    # Each thread produces N shards where N = num_shards / num_threads. For
    # instance, if num_shards = 128, and num_threads = 2, then the first thread
    # would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        
        for i in images_in_shard:
            image = images[i]
            sequence_example = _to_sequence_example(image, vocab)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

        if not counter % 1000:
            print("%s [thread %d]: Processed %d of %d items in thread batch." %
                  (datetime.now(), thread_index, counter, num_images_in_thread))
            sys.stdout.flush()

        writer.close()
        print("%s [thread %d]: Wrote %d image-caption pairs to %s" %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
          (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()


def _process_dataset(name, images, vocab, num_shards):
    """Processes a complete data set and saves it as a TFRecord.
    Args:
        name: Unique identifier specifying the dataset.
        images: List of ImageMetadata.
        vocab: A Vocabulary object.
        num_shards: Integer number of shards for the output files.
    """
    
    # Shuffle the ordering of images. Make the randomization repeatable.
    random.seed(12345)
    random.shuffle(images)

    # Break the images into num_threads batches. Batch i is defined as
    # images[ranges[i][0]:ranges[i][1]].
    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Launch a thread for each batch.
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in xrange(len(ranges)):
        args = (thread_index, ranges, name, images, vocab, num_shards)
        t = threading.Thread(target=_process_image_files, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d image-caption pairs in data set '%s'." %
          (datetime.now(), len(images), name))


def _process_caption(caption):
    """Processes a caption string into a list of tonenized words.
    Args:
        caption: A string caption.
    Returns:
        A list of strings; the tokenized caption.
    """
    tokenized_caption = [FLAGS.start_word]
    tokenized_caption.extend(nltk.tokenize.word_tokenize(caption.lower()))
    tokenized_caption.append(FLAGS.end_word)
    return tokenized_caption


def _load_and_process_metadata(captions_file, image_dir):
    """Loads image metadata from a JSON file and processes the captions.
    Args:
        captions_file: JSON file containing caption annotations.
        image_dir: Directory containing the image files.
    Returns:
        A list of ImageMetadata.
    """
    with tf.gfile.FastGFile(captions_file, "rb") as f:
        caption_data = json.load(f)

    # Extract the filenames.
    id_to_filename = [(x["id"], x["file_name"]) for x in caption_data["images"]]

    # Extract the captions. Each image_id is associated with multiple captions.
    id_to_captions = {}
    for annotation in caption_data["annotations"]:
        image_id = annotation["image_id"]
        caption = annotation["caption"]
        id_to_captions.setdefault(image_id, [])
        id_to_captions[image_id].append(caption)

    assert len(id_to_filename) == len(id_to_captions)
    assert set([x[0] for x in id_to_filename]) == set(id_to_captions.keys())
    print("Loaded caption metadata for %d images from %s" %
          (len(id_to_filename), captions_file))

    # Process the captions and combine the data into a list of ImageMetadata.
    print("Processing captions.")
    image_metadata = []
    num_captions = 0
    for image_id, base_filename in id_to_filename:
        filename = os.path.join(image_dir, base_filename)
        captions = [_process_caption(c) for c in id_to_captions[image_id]]
        image_metadata.append(ImageMetadata(image_id, filename, captions))
        num_captions += len(captions)
    # Break up each image into a separate entity for each caption.
    images = [ImageMetadata(image.image_id, image.filename, [caption])
              for image in image_metadata for caption in image.captions]
    print("Finished processing %d captions for %d images in %s" %
        (num_captions, len(id_to_filename), captions_file))
    
    return images

        
def _preextract_batch(batch_of_images, run_model_fn):
    """Runs the box extractor model on a single batch of images.
    Args:
        batch_of_images: list of (ImageMetadata, np.float32 shape [640, 640, 3]).
        run_model_fn: function that accepts np.float32 shape [batch_size, 640, 640, 3]
    Returns:
        A list of PreextractedMetadata.
    """

    # Extract the image and object features for a batch
    images, tensors = zip(*batch_of_images)
    boxes, scores = run_model_fn(np.stack(tensors))
    output_dataset = [PreextractedMetadata(
        image.image_id, image.filename, image.captions, 
        scores[j, :], boxes[j, :]) for j, image in enumerate(images)]
    
    return output_dataset


def _preextract_dataset(dataset_of_images, run_model_fn, BATCH_SIZE=32):
    """Runs the box extractor model on a single batch of images.
    Args:
        dataset_of_images: list of ImageMetadata.
        run_model_fn: function that accepts np.float32 shape [batch_size, 640, 640, 3]
    Returns:
        A list of PreextractedMetadata.
    """

    # Prepare batches to pass through the SSD Model
    batch_of_images, output_dataset = [], []
    decoder = ImageDecoder()
    for i, image in enumerate(dataset_of_images):
        with tf.gfile.FastGFile(image.filename, "rb") as f:
            encoded_image = f.read()
            
        try:
            batch_of_images += [[image, decoder.decode_jpeg(encoded_image)]]
        except (tf.errors.InvalidArgumentError, AssertionError):
            print("Skipping file with invalid JPEG data: %s" % image.filename)
            continue
            
        if len(batch_of_images) % BATCH_SIZE == 0:
            print("%s: Starting batch %d of %d on Box Extractor" %
                (datetime.now(), i // BATCH_SIZE, len(dataset_of_images) // BATCH_SIZE))
            output_dataset += _preextract_batch(batch_of_images, run_model_fn)
            batch_of_images = []
            print("%s: Finished running batch %d of %d on Box Extractor" %
                (datetime.now(), i // BATCH_SIZE, len(dataset_of_images) // BATCH_SIZE))
            
    if len(batch_of_images) != 0:
            output_dataset += _preextract_batch(batch_of_images, run_model_fn)
            batch_of_images = []
    print("%s: Finished all %d batches on Box Extractor" %
        (datetime.now(), len(dataset_of_images) // BATCH_SIZE))
    
    return output_dataset


def main(unused_argv):
    def _is_valid_num_shards(num_shards):
        """Returns True if num_shards is compatible with FLAGS.num_threads."""
        return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

    assert _is_valid_num_shards(FLAGS.train_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
    assert _is_valid_num_shards(FLAGS.val_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")
    assert _is_valid_num_shards(FLAGS.test_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.test_shards")

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    # Load image metadata from caption files.
    mscoco_train_dataset = _load_and_process_metadata(FLAGS.train_captions_file, FLAGS.train_image_dir)
    mscoco_val_dataset = _load_and_process_metadata(FLAGS.val_captions_file, FLAGS.val_image_dir)

    # Redistribute the MSCOCO data as follows:
    #   train_dataset = 99% of mscoco_train_dataset
    #   val_dataset = 1% of mscoco_train_dataset (for validation during training).
    #   test_dataset = 100% of mscoco_val_dataset (for final evaluation).
    train_cutoff = int(0.99 * len(mscoco_train_dataset))
    train_dataset = mscoco_train_dataset[:train_cutoff]
    val_dataset = mscoco_train_dataset[train_cutoff:]
    test_dataset = mscoco_val_dataset

    # Create vocabulary from the glove embeddings.
    vocab, _ = load_glove(vocab_size=FLAGS.vocab_size, embedding_size=FLAGS.embedding_size)

    # Create the model to extract image features
    box_extractor = BoxExtractor(get_faster_rcnn_config(), trainable=False)
    image_tensor = tf.placeholder(tf.float32, name='image_tensor', shape=[None, None, None, 3])
    boxes, scores, _cropped_images = box_extractor(image_tensor)

    with tf.Session() as sess:

        saver = tf.train.Saver(var_list=box_extractor.variables)
        saver.restore(sess, get_faster_rcnn_checkpoint())
        def run_model_fn(images):
            return sess.run([boxes, scores], feed_dict={image_tensor: images})
        
        train_dataset = _preextract_dataset(train_dataset, run_model_fn)
        val_dataset = _preextract_dataset(val_dataset, run_model_fn)
        test_dataset = _preextract_dataset(test_dataset, run_model_fn)

    _process_dataset("train", train_dataset, vocab, FLAGS.train_shards)
    _process_dataset("val", val_dataset, vocab, FLAGS.val_shards)
    _process_dataset("test", test_dataset, vocab, FLAGS.test_shards)


if __name__ == "__main__":
    tf.app.run()