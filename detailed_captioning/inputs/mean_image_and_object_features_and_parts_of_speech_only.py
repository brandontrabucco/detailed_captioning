'''Author: Brandon Trabucco, Copyright 2019
Load the MSCOCO dataset serialized to tensorflow sequence examples.'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


def _load_dataset_from_tf_records(mode, is_mini):
    assert(mode in ["train", "eval", "test"])
    if mode == "train":
        input_file_pattern = "data/coco{0}/train-?????-of-?????".format("_mini" if is_mini else "")
    if mode == "eval":
        input_file_pattern = "data/coco{0}/val-?????-of-?????".format("_mini" if is_mini else "")
    if mode == "test":
        input_file_pattern = "data/coco{0}/test-?????-of-?????".format("_mini" if is_mini else "")
    data_files = tf.data.Dataset.list_files(input_file_pattern)
    return data_files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=4, sloppy=True))


def _process_tf_record_proto(serialized_proto):
    context, sequence = tf.parse_single_sequence_example(
        serialized_proto,
        context_features = {
            "image/image_id": tf.FixedLenFeature([], dtype=tf.int64)},
        sequence_features = {
            "image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "image/parts_of_speech_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "image/image_features": tf.FixedLenSequenceFeature([], dtype=tf.float32),
            "image/object_features": tf.FixedLenSequenceFeature([], dtype=tf.float32)})
    image_id, caption = (
        context["image/image_id"], sequence["image/caption_ids"])
    image_features, parts_of_speech = (
        sequence["image/image_features"], sequence["image/parts_of_speech_ids"])
    object_features = sequence["image/object_features"]
    return {"image_id": image_id, "caption": caption, 
            "image_features": image_features, "object_features": object_features, 
            "parts_of_speech": parts_of_speech}


def _mask_and_slice(x):
    image_id, caption, image_features, object_features, parts_of_speech = (
        x["image_id"], x["caption"], x["image_features"], x["object_features"], x["parts_of_speech"])
    caption_length = tf.shape(caption)[0]
    input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)
    input_seq = tf.slice(caption, [0], input_length)
    target_seq = tf.slice(caption, [1], input_length)
    indicator = tf.ones(input_length, dtype=tf.int32)
    pos_caption_length = tf.shape(parts_of_speech)[0]
    pos_input_length = tf.expand_dims(tf.subtract(pos_caption_length, 1), 0)
    pos_input_seq = tf.slice(parts_of_speech, [0], input_length)
    pos_target_seq = tf.slice(parts_of_speech, [1], input_length)
    pos_indicator = tf.ones(pos_input_length, dtype=tf.int32)
    return {"image_id": image_id, "image_features": image_features, "object_features": object_features, 
            "input_seq": input_seq, "target_seq": target_seq, 
            "indicator": indicator,
            "pos_input_seq": pos_input_seq, "pos_target_seq": pos_target_seq,
            "pos_indicator": pos_indicator}


def _prepare_final_batch(x):
    (image_id, image_features, object_features, input_seq, target_seq, indicator, 
     pos_input_seq, pos_target_seq, pos_indicator) = (
        x["image_id"], x["image_features"], x["object_features"], 
        x["input_seq"], x["target_seq"], x["indicator"], 
        x["pos_input_seq"], x["pos_target_seq"], x["pos_indicator"] )
    image_id = tf.cast(image_id, tf.int32)
    image_features = tf.cast(image_features, tf.float32)
    target_shape = [tf.shape(image_features)[0], 7, 7, 2048]
    image_features = tf.reduce_mean(tf.reshape(image_features, target_shape), [1, 2])
    object_features = tf.cast(object_features, tf.float32)
    object_shape = [tf.shape(object_features)[0], 8, 2048]
    object_features = tf.reshape(object_features, object_shape)
    input_seq = tf.cast(input_seq, tf.int32)
    target_seq = tf.cast(target_seq, tf.int32)
    indicator = tf.cast(indicator, tf.float32)
    pos_input_seq = tf.cast(pos_input_seq, tf.int32)
    pos_target_seq = tf.cast(pos_target_seq, tf.int32)
    pos_indicator = tf.cast(pos_indicator, tf.float32)
    return {"image_id": image_id, 
            "image_features": image_features, "object_features": object_features, 
            "input_seq": input_seq, "target_seq": target_seq, 
            "indicator": indicator,
            "pos_input_seq": pos_input_seq, "pos_target_seq": pos_target_seq,
            "pos_indicator": pos_indicator}
    

def import_mscoco(mode="train", is_mini=True, batch_size=100, num_epochs=1):
    is_training = (mode == "train")
    dataset = _load_dataset_from_tf_records(mode, is_mini)
    dataset = dataset.map(_process_tf_record_proto, num_parallel_calls=4)
    dataset = dataset.map(_mask_and_slice, num_parallel_calls=4)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000, count=num_epochs))
    padded_shapes = {"image_id": [], 
                     "image_features": [7 * 7 * 2048], "object_features": [8 * 2048], 
                     "input_seq": [None], "target_seq": [None], "indicator": [None], 
                     "pos_input_seq": [None], "pos_target_seq": [None], "pos_indicator": [None]}
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True)
    dataset = dataset.map(_prepare_final_batch, num_parallel_calls=4)
    dataset = dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0", buffer_size=2))
    iterator = dataset.make_one_shot_iterator()
    x = iterator.get_next()
    (image_id, image_features, object_features, input_seq, target_seq, indicator, 
     pos_input_seq, pos_target_seq, pos_indicator) = (
        x["image_id"], x["image_features"], x["object_features"], 
        x["input_seq"], x["target_seq"], x["indicator"], 
        x["pos_input_seq"], x["pos_target_seq"], x["pos_indicator"])
    return (image_id, image_features, object_features, input_seq, target_seq, indicator, 
        pos_input_seq, pos_target_seq, pos_indicator)
