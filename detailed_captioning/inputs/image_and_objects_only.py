'''Author: Brandon Trabucco, Copyright 2019
Load the MSCOCO dataset serialized to tensorflow sequence examples.'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


def _repeat_elements(num_elements, num_repeats):
    # Tensor: [[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, ...]]
    return tf.reshape(tf.tile(tf.expand_dims(tf.range(num_elements), 1), 
                              [1, num_repeats]), [-1])


def _load_dataset_from_tf_records(mode):
    assert(mode in ["train", "eval", "test"])
    if mode == "train":
        input_file_pattern = "data/coco_boxes/train-?????-of-?????"
    if mode == "eval":
        input_file_pattern = "data/coco_boxes/val-?????-of-?????"
    if mode == "test":
        input_file_pattern = "data/coco_boxes/test-?????-of-?????"
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
            "image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "image/boxes": tf.FixedLenSequenceFeature([], dtype=tf.float32)})
    image, image_id, caption, boxes = (
        context["image/data"], context["image/image_id"], 
        sequence["image/caption_ids"], sequence["image/boxes"])
    return {"image": image, "image_id": image_id, "caption": caption, "boxes": boxes}


def _mask_and_slice(x):
    image, image_id, caption, boxes = x["image"], x["image_id"], x["caption"], x["boxes"]
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, size=[256, 256], method=tf.image.ResizeMethod.BILINEAR)
    caption_length = tf.shape(caption)[0]
    input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)
    input_seq = tf.slice(caption, [0], input_length)
    target_seq = tf.slice(caption, [1], input_length)
    indicator = tf.ones(input_length, dtype=tf.int32)
    return {"image_id": image_id, "input_seq": input_seq, 
            "target_seq": target_seq, "indicator": indicator, 
            "image": image, "boxes": boxes}


def _crop_object_boxes(x):
    image_id, input_seq, target_seq, indicator, image, boxes = (
        x["image_id"], x["input_seq"], x["target_seq"], x["indicator"], 
        x["image"], x["boxes"])
    batch_size = tf.shape(boxes)[0]
    boxes = tf.reshape(boxes, [batch_size, 100, 4])
    cropped_image = tf.image.crop_and_resize(image, tf.reshape(boxes, [-1, 4]), 
        _repeat_elements(batch_size, 100), [256, 256])
    return {"image_id": image_id, "input_seq": input_seq, 
            "target_seq": target_seq, "indicator": indicator, 
            "image": image, "cropped_image": cropped_image}


def _random_distort(x):
    image_id, input_seq, target_seq, indicator, image, cropped_image = (
        x["image_id"], x["input_seq"], x["target_seq"], x["indicator"], 
        x["image"], x["cropped_image"])
    image = tf.random_crop(image, [tf.shape(image)[0], 224, 224, 3])
    image = tf.image.random_flip_left_right(image / 255.)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.032)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.clip_by_value(image, 0.0, 1.0) * 255.
    # Do the same for the cropped regions
    cropped_image = tf.random_crop(cropped_image, [tf.shape(cropped_image)[0], 224, 224, 3])
    cropped_image = tf.image.random_flip_left_right(cropped_image / 255.)
    cropped_image = tf.image.random_brightness(cropped_image, max_delta=32. / 255.)
    cropped_image = tf.image.random_saturation(cropped_image, lower=0.5, upper=1.5)
    cropped_image = tf.image.random_hue(cropped_image, max_delta=0.032)
    cropped_image = tf.image.random_contrast(cropped_image, lower=0.5, upper=1.5)
    cropped_image = tf.clip_by_value(cropped_image, 0.0, 1.0) * 255.
    return {"image_id": image_id, "input_seq": input_seq, 
            "target_seq": target_seq, "indicator": indicator, 
            "image": image, "cropped_image": cropped_image}


def _crop_or_pad(x):
    image_id, input_seq, target_seq, indicator, image, cropped_image = (
        x["image_id"], x["input_seq"], x["target_seq"], x["indicator"], 
        x["image"], x["cropped_image"])
    image = tf.image.resize_images(image, size=[224, 224], 
        method=tf.image.ResizeMethod.BILINEAR)
    cropped_image = tf.image.resize_images(cropped_image, size=[224, 224], 
        method=tf.image.ResizeMethod.BILINEAR)
    return {"image_id": image_id, "input_seq": input_seq, 
            "target_seq": target_seq, "indicator": indicator, 
            "image": image, "cropped_image": cropped_image}


def _prepare_final_batch(x):
    image_id, input_seq, target_seq, indicator, image, cropped_image = (
        x["image_id"], x["input_seq"], x["target_seq"], x["indicator"], 
        x["image"], x["cropped_image"])
    image_id = tf.cast(image_id, tf.int32)
    input_seq = tf.cast(input_seq, tf.int32)
    target_seq = tf.cast(target_seq, tf.int32)
    indicator = tf.cast(indicator, tf.float32)
    image = tf.cast(image, tf.float32)
    cropped_image = tf.cast(cropped_image, tf.float32)
    return {"image_id": image_id, "input_seq": input_seq, 
            "target_seq": target_seq, "indicator": indicator, 
            "image": image, "cropped_image": cropped_image}
    

def import_mscoco(mode="train", batch_size=100, num_epochs=1):
    is_training = (mode == "train")
    dataset = _load_dataset_from_tf_records(mode)
    dataset = dataset.map(_process_tf_record_proto, num_parallel_calls=4)
    dataset = dataset.map(_mask_and_slice, num_parallel_calls=4)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000, count=num_epochs))
    padded_shapes = {"image_id": [], "input_seq": [None], 
                     "target_seq": [None], "indicator": [None], 
                     "image": [256, 256, 3], "boxes": [400]}
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True)
    dataset = dataset.map(_crop_object_boxes, num_parallel_calls=4)
    if is_training:
        dataset = dataset.map(_random_distort, num_parallel_calls=4)
    else:
        dataset = dataset.map(_crop_or_pad, num_parallel_calls=4)
    dataset = dataset.map(_prepare_final_batch, num_parallel_calls=4)
    dataset = dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0", buffer_size=2))
    iterator = dataset.make_one_shot_iterator()
    x = iterator.get_next()
    image_id, input_seq, target_seq, indicator, image, cropped_image = (
        x["image_id"], x["input_seq"], x["target_seq"], x["indicator"], x["image"], x["cropped_image"])
    return image_id, image, cropped_image, input_seq, target_seq, indicator
