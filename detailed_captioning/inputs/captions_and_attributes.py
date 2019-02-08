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
            "image/image_id": tf.FixedLenFeature([], dtype=tf.int64)
        },
        sequence_features = {
            "image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "image/mean_image_features": tf.FixedLenSequenceFeature([], dtype=tf.float32),
            "image/mean_object_features": tf.FixedLenSequenceFeature([], dtype=tf.float32),
            "image/attributes": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        })
    caption = sequence["image/caption_ids"]
    image_id = context["image/image_id"]
    image_features = sequence["image/mean_image_features"]
    object_features = sequence["image/mean_object_features"]
    attributes = sequence["image/attributes"]
    spt = tf.SparseTensor(tf.expand_dims(attributes, 1),
                         tf.ones(tf.shape(attributes)[0]),
                         [1000])
    attributes = tf.sparse_tensor_to_dense(spt)
    return {"image_id": image_id, "caption": caption, 
            "image_features": image_features, "object_features": object_features, 
            "attributes": attributes}


def _mask_and_slice(x):
    image_id, caption, image_features, object_features, attributes = (
        x["image_id"], x["caption"], x["image_features"], x["object_features"], x["attributes"])
    caption_length = tf.shape(caption)[0]
    input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)
    input_seq = tf.slice(caption, [0], input_length)
    target_seq = tf.slice(caption, [1], input_length)
    indicator = tf.ones(input_length, dtype=tf.int32)
    return {"image_id": image_id, 
            "image_features": image_features, "object_features": object_features, 
            "input_seq": input_seq, "target_seq": target_seq, 
            "indicator": indicator,
            "attributes": attributes}


def _prepare_final_batch(x):
    image_id, image_features, object_features, input_seq, target_seq, indicator, attributes = (
        x["image_id"], x["image_features"], x["object_features"], 
        x["input_seq"], x["target_seq"], x["indicator"], 
        x["attributes"])
    target_shape = [tf.shape(image_features)[0], 2048]
    image_features = tf.reshape(image_features, target_shape)
    target_shape = [tf.shape(object_features)[0], 8, 2048]
    object_features = tf.reshape(object_features, target_shape)
    image_id = tf.cast(image_id, tf.int32)
    image_features = tf.cast(image_features, tf.float32)
    object_features = tf.cast(object_features, tf.float32)
    attributes = tf.cast(attributes, tf.float32)
    return {"image_id": image_id, 
            "image_features": image_features, "object_features": object_features, 
            "input_seq": input_seq, "target_seq": target_seq, 
            "indicator": indicator,
            "attributes": attributes}
    

def import_mscoco(mode="train", is_mini=True, batch_size=100, num_epochs=1):
    is_training = (mode == "train")
    dataset = _load_dataset_from_tf_records(mode, is_mini)
    dataset = dataset.map(_process_tf_record_proto, num_parallel_calls=4)
    dataset = dataset.map(_mask_and_slice, num_parallel_calls=4)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000, count=num_epochs))
    padded_shapes = {"image_id": [], 
            "image_features": [2048], "object_features": [8 * 2048], 
            "input_seq": [None], "target_seq": [None], 
            "indicator": [None],
            "attributes": [1000]}
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True)
    dataset = dataset.map(_prepare_final_batch, num_parallel_calls=4)
    dataset = dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0", buffer_size=2))
    iterator = dataset.make_one_shot_iterator()
    x = iterator.get_next()
    image_id, image_features, object_features, input_seq, target_seq, indicator, attributes = (
        x["image_id"], x["image_features"], x["object_features"], 
        x["input_seq"], x["target_seq"], x["indicator"], 
        x["attributes"])
    return image_id, image_features, object_features, input_seq, target_seq, indicator, attributes
