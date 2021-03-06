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
            "image/data": tf.FixedLenFeature([], dtype=tf.string),
            "image/image_id": tf.FixedLenFeature([], dtype=tf.int64)},
        sequence_features = {
            "image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64)})
    image, image_id, caption = (
        context["image/data"], context["image/image_id"], sequence["image/caption_ids"])
    return {"image": image, "image_id": image_id, "caption": caption}


def _mask_and_slice(x):
    image, image_id, caption = x["image"], x["image_id"], x["caption"]
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, size=[320, 320], method=tf.image.ResizeMethod.BILINEAR)
    caption_length = tf.shape(caption)[0]
    input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)
    input_seq = tf.slice(caption, [0], input_length)
    target_seq = tf.slice(caption, [1], input_length)
    indicator = tf.ones(input_length, dtype=tf.int32)
    return {"image_id": image_id, "input_seq": input_seq, 
            "target_seq": target_seq, "indicator": indicator, 
            "image": image}


def _random_distort(x):
    image, image_id, caption = x["image"], x["image_id"], x["caption"]
    image = tf.random_crop(image, [tf.shape(image)[0], 224, 224, 3])
    image = tf.image.random_flip_left_right(image / 255.)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.032)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.clip_by_value(image, 0.0, 1.0) * 255.
    return {"image": image, "image_id": image_id, "caption": caption}


def _crop_or_pad(x):
    image, image_id, caption = x["image"], x["image_id"], x["caption"]
    image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
    return {"image": image, "image_id": image_id, "caption": caption}


def _prepare_final_batch(x):
    image_id, input_seq, target_seq, indicator, image = (
        x["image_id"], x["input_seq"], x["target_seq"], x["indicator"], 
        x["image"])
    image_id = tf.cast(image_id, tf.int32)
    input_seq = tf.cast(input_seq, tf.int32)
    target_seq = tf.cast(target_seq, tf.int32)
    indicator = tf.cast(indicator, tf.float32)
    image = tf.cast(image, tf.float32)
    return {"image_id": image_id, "input_seq": input_seq, 
            "target_seq": target_seq, "indicator": indicator, 
            "image": image}
    

def import_mscoco(mode="train", is_mini=True, batch_size=100, num_epochs=1):
    is_training = (mode == "train")
    dataset = _load_dataset_from_tf_records(mode, is_mini)
    dataset = dataset.map(_process_tf_record_proto, num_parallel_calls=4)
    dataset = dataset.map(_mask_and_slice, num_parallel_calls=4:
    if is_training:
        dataset = dataset.map(_random_distort, num_parallel_calls=4)
    else:
        dataset = dataset.map(_crop_or_pad, num_parallel_calls=4)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000, count=num_epochs))
    padded_shapes = {"image_id": [], "input_seq": [None], 
                     "target_seq": [None], "indicator": [None], 
                     "image": [224, 224, 3]}
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True)
    dataset = dataset.map(_prepare_final_batch, num_parallel_calls=4)
    dataset = dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0", buffer_size=2))
    iterator = dataset.make_one_shot_iterator()
    x = iterator.get_next()
    image_id, input_seq, target_seq, indicator, image = (
        x["image_id"], x["input_seq"], x["target_seq"], x["indicator"], x["image"])
    return image_id, image, input_seq, target_seq, indicator
