'''Author: Brandon Trabucco, Copyright 2019
Builds TF Records for the MSCOCO 2017 dataset

NOTE: this only extracts mean image features and mean object features, not spatial features.

'''


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
from detailed_captioning.utils import get_visual_attributes
from detailed_captioning.utils import load_glove
from detailed_captioning.layers.box_extractor import BoxExtractor
from detailed_captioning.utils import get_faster_rcnn_config
from detailed_captioning.utils import get_faster_rcnn_checkpoint


tf.logging.set_verbosity(tf.logging.INFO)


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

tf.flags.DEFINE_integer("vocab_size", 100000, "")
tf.flags.DEFINE_integer("image_height", 224, "")
tf.flags.DEFINE_integer("image_width", 224, "")
tf.flags.DEFINE_integer("batch_size", 32, "")

FLAGS = tf.flags.FLAGS

ImageMetadata = namedtuple("ImageMetadata",
                           ["image_id", "filename", "captions"])

PreextractedMetadata = namedtuple("PreextractedMetadata",
                           ["image_id", "filename", "caption", 
                            "attributes", "boxes"])

class Extractor(object):
    """Helper class for extracting boxes from images."""
    
    def __init__(self):
        """Creates handles to the TensorFlow computational graph."""
        # TensorFlow ops for JPEG decoding.
        self.encoded_jpeg = tf.placeholder(dtype=tf.string)
        self.decoded_jpeg = tf.image.decode_jpeg(self.encoded_jpeg, channels=3)
        self.decoded_jpeg = tf.image.resize_images(self.decoded_jpeg, [
            FLAGS.image_height, FLAGS.image_width])
        
        # Create the model to extract image boxes
        self.box_extractor = BoxExtractor(get_faster_rcnn_config(), trainable=False)
        self.image_tensor = tf.placeholder(tf.float32, name='image_tensor', shape=[None, 
            FLAGS.image_height, FLAGS.image_width, 3])
        self.boxes, self.scores, self.cropped_images = self.box_extractor(self.image_tensor)
        
        # Create a single TensorFlow Session for all image decoding calls.
        self.sess = tf.Session()
        self.rcnn_saver = tf.train.Saver(var_list=self.box_extractor.variables)
        self.rcnn_saver.restore(self.sess, get_faster_rcnn_checkpoint())
        self.lock = threading.Lock()
        self.attribute_map = get_visual_attributes()
        
    def extract(self, list_of_image_metadata):
        """Extracts a batch of image boxes.
        Args: 
            list_of_image_metadata: a list of ImageMetadata objects.
        Returns: 
            a list of PreextractedMetadata objects."""
        encoded_jpegs = []
        for x in list_of_image_metadata:
            with tf.gfile.FastGFile(x.filename, "rb") as f:
                encoded_image = f.read()
                encoded_jpegs.append(encoded_image)
                
        images = np.stack([self.sess.run(self.decoded_jpeg, feed_dict={
            self.encoded_jpeg: x}) for x in encoded_jpegs], axis=0)
        self.lock.acquire()
        result = self.sess.run(self.boxes, feed_dict={self.image_tensor: images})
        self.lock.release()
        
        list_of_preextracted_metadata = []
        for i in range(len(list_of_image_metadata)):
            image_meta = list_of_image_metadata[i]
            list_of_preextracted_metadata.append(
                PreextractedMetadata(
                    image_id=image_meta.image_id,
                    filename=image_meta.filename,
                    caption=image_meta.captions[0],
                    attributes=self.attribute_map.sentence_to_attributes(
                        image_meta.captions[0]),
                    boxes=result[i, ...]))
            
        return list_of_preextracted_metadata


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
        image: An PreextractedMetadata object.
        vocab: A Vocabulary object.
    Returns: 
        A SequenceExample proto."""
    with tf.gfile.FastGFile(image.filename, "rb") as f:
        encoded_image = f.read()
    sequence_example = tf.train.SequenceExample(
        context=tf.train.Features(feature={
            "image/image_id": _int64_feature(image.image_id),
            "image/data": _bytes_feature(encoded_image)}), 
        feature_lists=tf.train.FeatureLists(feature_list={
            "image/caption_ids": _int64_feature_list(
                [vocab.start_id] + vocab.word_to_id(image.caption) + [vocab.end_id]),
            "image/attributes": _int64_feature_list(image.attributes),
            "image/boxes": _float_feature_list(image.boxes.flatten().tolist())}))
    return sequence_example


def _preextract_dataset(dataset_of_images, dataset_extractor):
    """Runs the box extractor model on a single batch of images.
    Args:
        dataset_of_images: list of ImageMetadata.
        dataset_extractor: an instance of Extractor object, extracts image metadata.
    Returns:
        A list of PreextractedMetadata."""

    # Prepare batches to pass through the Faster RCNN Model
    BATCH_SIZE = FLAGS.batch_size
    batch_of_images, output_dataset = [], []
    for i, image in enumerate(dataset_of_images):
            
        batch_of_images.append(image)
            
        if len(batch_of_images) % BATCH_SIZE == 0:
            print("%s: Starting batch %d of %d on Extractor" %
                (datetime.now(), i // BATCH_SIZE, len(dataset_of_images) // BATCH_SIZE))
            try:
                output_dataset += dataset_extractor.extract(batch_of_images)
            except (tf.errors.InvalidArgumentError, AssertionError):
                print("Skipping file with invalid JPEG data: %s" % image.filename)
                continue
            batch_of_images = []
            print("%s: Finished running batch %d of %d on Extractor" %
                (datetime.now(), i // BATCH_SIZE, len(dataset_of_images) // BATCH_SIZE))
            
    if len(batch_of_images) != 0:
        try:
            output_dataset += dataset_extractor.extract(batch_of_images)
        except (tf.errors.InvalidArgumentError, AssertionError):
            print("Skipping file with invalid JPEG data: %s" % image.filename)
    
    print("%s: Finished all %d batches on Extractor" %
        (datetime.now(), len(dataset_of_images) // BATCH_SIZE))
    
    return output_dataset
        

def _process_image_files(thread_index, ranges, name, images, vocab, num_shards, 
                         dataset_extractor):
    """Processes and saves a subset of images as TFRecord files in one thread.
    Args:
        thread_index: Integer thread identifier within [0, len(ranges)].
        ranges: A list of pairs of integers specifying the ranges of the dataset to
                process in parallel.
        name: Unique identifier specifying the dataset.
        images: List of ImageMetadata.
        vocab: A Vocabulary object.
        num_shards: Integer number of shards for the output files.
        dataset_extractor: an instance of Extractor object, extracts image metadata."""
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
        
        these_images = images[shard_ranges[s]:shard_ranges[s + 1]]
        these_images = _preextract_dataset(these_images, dataset_extractor)
        
        for image in these_images:
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


def _process_dataset(name, images, vocab, num_shards, dataset_extractor):
    """Processes a complete data set and saves it as a TFRecord.
    Args:
        name: Unique identifier specifying the dataset.
        images: List of ImageMetadata.
        vocab: A Vocabulary object.
        num_shards: Integer number of shards for the output files.
        dataset_extractor: an instance of Extractor object, extracts image metadata."""
    
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
        args = (thread_index, ranges, name, images, vocab, num_shards, dataset_extractor)
        t = threading.Thread(target=_process_image_files, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d image-caption pairs in data set '%s'." %
          (datetime.now(), len(images), name))


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
        captions = [nltk.tokenize.word_tokenize(c.lower()) for c in id_to_captions[image_id]]
        image_metadata.append(ImageMetadata(image_id, filename, captions))
        num_captions += len(captions)
    # Break up each image into a separate entity for each caption.
    images = [ImageMetadata(image.image_id, image.filename, [caption])
              for image in image_metadata for caption in image.captions]
    print("Finished processing %d captions for %d images in %s" %
        (num_captions, len(id_to_filename), captions_file))
    
    return images


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
        
    # Create vocabulary from the glove embeddings.
    vocab, _ = load_glove(vocab_size=FLAGS.vocab_size, embedding_size=50)

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
    
    dataset_extractor = Extractor()
    _process_dataset("train", train_dataset, vocab, FLAGS.train_shards, dataset_extractor)
    _process_dataset("val", val_dataset, vocab, FLAGS.val_shards, dataset_extractor)
    _process_dataset("test", test_dataset, vocab, FLAGS.test_shards, dataset_extractor)


if __name__ == "__main__":
    tf.app.run()