'''Author: Brandon Trabucco, Copyright 2019
Load the MSCOCO dataset serialized to tensorflow sequence examples.'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


def _process_tf_record_proto(serialized_proto):
    context, sequence = tf.parse_single_sequence_example(
        serialized_proto,
        context_features = {
            "image/image_id": tf.FixedLenFeature([], dtype=tf.int64)},
        sequence_features = {
            "image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "image/image_features": tf.FixedLenSequenceFeature([], dtype=tf.float32),
            "image/image_features_shape": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "image/object_features": tf.FixedLenSequenceFeature([], dtype=tf.float32),
            "image/object_features_shape": tf.FixedLenSequenceFeature([], dtype=tf.int64)})
    image_id, caption = (
        context["image/image_id"], sequence["image/caption_ids"])
    image_features = tf.reshape(sequence["image/image_features"], sequence["image/image_features_shape"])
    image_features = tf.reduce_mean(image_features, [1, 2])
    object_features = tf.reshape(sequence["image/object_features"], sequence["image/object_features_shape"])
    return {"image_id": image_id, "caption": caption, 
            "image_features": image_features, "object_features": object_features}


def _mask_and_slice(x):
    image_id, caption, image_features, object_features = (
        x["image_id"], x["caption"], x["image_features"], x["object_features"])
    caption_length = tf.shape(caption)[0]
    input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)
    input_seq = tf.slice(caption, [0], input_length)
    target_seq = tf.slice(caption, [1], input_length)
    indicator = tf.ones(input_length, dtype=tf.int32)
    return {"image_id": image_id, "input_seq": input_seq, 
            "target_seq": target_seq, "indicator": indicator, 
            "image_features": image_features, "object_features": object_features}


def _convert_dtype(x):
    image_id, input_seq, target_seq, indicator, image_features, object_features = (
        x["image_id"], x["input_seq"], x["target_seq"], x["indicator"], 
        x["image_features"], x["object_features"])
    image_id = tf.cast(image_id, tf.int32)
    input_seq = tf.cast(input_seq, tf.int32)
    target_seq = tf.cast(target_seq, tf.int32)
    indicator = tf.cast(indicator, tf.float32)
    return {"image_id": image_id, "input_seq": input_seq, 
            "target_seq": target_seq, "indicator": indicator, 
            "image_features": image_features, "object_features": object_features}


def _load_dataset_from_tf_records(mode, is_mini):
    assert(mode in ["train", "eval", "test"])
    if mode == "train":
        input_file_pattern = "data/coco{0}/train-?????-of-?????".format("_mini" if is_mini else "")
    if mode == "eval":
        input_file_pattern = "data/coco{0}/val-?????-of-?????".format("_mini" if is_mini else "")
    if mode == "test":
        input_file_pattern = "data/coco{0}/test-?????-of-?????".format("_mini" if is_mini else "")
    data_files = []
    for pattern in input_file_pattern.split(","):
        data_files.extend(tf.gfile.Glob(pattern))
    print("Found {0} files matching {1}".format(len(data_files), input_file_pattern))
    return tf.data.TFRecordDataset(data_files)
    
    
def _apply_dataset_transformations(dataset, is_training):
    dataset = dataset.map(_process_tf_record_proto)
    dataset = dataset.map(_mask_and_slice)
    return dataset.map(_convert_dtype)


def import_mscoco(mode="train", shuffle=True, batch_size=100, num_epochs=1, num_boxes=8, is_mini=True):
    is_training = (mode == "train")
    dataset = _load_dataset_from_tf_records(mode, is_mini)
    dataset = _apply_dataset_transformations(dataset, is_training)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    padded_shapes = {"image_id": [], "input_seq": [None], 
                     "target_seq": [None], "indicator": [None], 
                     "image_features": [7, 7, 2048], "object_features": [num_boxes, 2048]}
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    x = iterator.get_next()
    image_id, input_seq, target_seq, indicator, image_features, object_features = (
        x["image_id"], x["input_seq"], x["target_seq"], x["indicator"], 
        x["image_features"], x["object_features"])
    return image_id, image_features, object_features, input_seq, target_seq, indicator
    