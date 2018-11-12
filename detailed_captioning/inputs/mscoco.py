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
            "image/data": tf.FixedLenFeature([], dtype=tf.string),
            "image/image_id": tf.FixedLenFeature([], dtype=tf.int64)},
        sequence_features = {
            "image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "image/image_features": tf.FixedLenSequenceFeature([], dtype=tf.float32),
            "image/image_features_shape": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "image/object_features": tf.FixedLenSequenceFeature([], dtype=tf.float32),
            "image/object_features_shape": tf.FixedLenSequenceFeature([], dtype=tf.int64)})
    image, image_id, caption = (
        context["image/data"], context["image/image_id"], sequence["image/caption_ids"])
    image_features = tf.reshape(sequence["image/image_features"], sequence["image/image_features_shape"])
    object_features = tf.reshape(sequence["image/object_features"], sequence["image/object_features_shape"])
    return {"image": image, "image_id": image_id, "caption": caption, 
            "image_features": image_features, "object_features": object_features}


def _decode_and_resize_image(x):
    image, image_id, caption, image_features, object_features = (
        x["image"], x["image_id"], x["caption"], x["image_features"], x["object_features"])
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, size=[320, 320], 
                                   method=tf.image.ResizeMethod.BILINEAR)
    return {"image": image, "image_id": image_id, "caption": caption, 
            "image_features": image_features, "object_features": object_features}
    

def _random_distort(x):
    image, image_id, caption, image_features, object_features = (
        x["image"], x["image_id"], x["caption"], x["image_features"], x["object_features"])
    image = tf.random_crop(image, [224, 224, 3])
    image = tf.image.random_flip_left_right(image / 255.)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.032)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.clip_by_value(image, 0.0, 1.0) * 255.
    return {"image": image, "image_id": image_id, "caption": caption, 
            "image_features": image_features, "object_features": object_features}


def _crop_or_pad(x):
    image, image_id, caption, image_features, object_features = (
        x["image"], x["image_id"], x["caption"], x["image_features"], x["object_features"])
    image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
    return {"image": image, "image_id": image_id, "caption": caption, 
            "image_features": image_features, "object_features": object_features}


def _mask_and_slice(x):
    image, image_id, caption, image_features, object_features = (
        x["image"], x["image_id"], x["caption"], x["image_features"], x["object_features"])
    caption_length = tf.shape(caption)[0]
    input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)
    input_seq = tf.slice(caption, [0], input_length)
    target_seq = tf.slice(caption, [1], input_length)
    indicator = tf.ones(input_length, dtype=tf.int32)
    return {"image": image, "image_id": image_id, "input_seq": input_seq, 
            "target_seq": target_seq, "indicator": indicator, 
            "image_features": image_features, "object_features": object_features}


def _convert_dtype(x):
    image, image_id, input_seq, target_seq, indicator, image_features, object_features = (
        x["image"], x["image_id"], x["input_seq"], x["target_seq"], x["indicator"], 
        x["image_features"], x["object_features"])
    image = tf.cast(image, tf.float32)
    image_id = tf.cast(image_id, tf.int32)
    input_seq = tf.cast(input_seq, tf.int32)
    target_seq = tf.cast(target_seq, tf.int32)
    indicator = tf.cast(indicator, tf.float32)
    return {"image": image, "image_id": image_id, "input_seq": input_seq, 
            "target_seq": target_seq, "indicator": indicator, 
            "image_features": image_features, "object_features": object_features}


def _load_dataset_from_tf_records(is_training, is_mini):
    if is_training:
        input_file_pattern = "data/coco{0}/train-?????-of-?????".format("_mini" if is_mini else "")
    if not is_training:
        input_file_pattern = "data/coco{0}/val-?????-of-?????".format("_mini" if is_mini else "")
    data_files = []
    for pattern in input_file_pattern.split(","):
        data_files.extend(tf.gfile.Glob(pattern))
    print("Found {0} files matching {1}".format(len(data_files), input_file_pattern))
    return tf.data.TFRecordDataset(data_files)
    
    
def _apply_dataset_transformations(dataset, is_training):
    dataset = dataset.map(_process_tf_record_proto)
    dataset = dataset.map(_decode_and_resize_image)
    if is_training:
        dataset = dataset.map(_random_distort)
    if not is_training:
        dataset = dataset.map(_crop_or_pad)
    dataset = dataset.map(_mask_and_slice)
    return dataset.map(_convert_dtype)


def import_mscoco(is_training=True, batch_size=32, num_epochs=1, num_boxes=8, is_mini=True):
    dataset = _load_dataset_from_tf_records(is_training, is_mini)
    dataset = _apply_dataset_transformations(dataset, is_training)
    dataset = dataset.shuffle(buffer_size=1000)
    padded_shapes = {"image": [224, 224, 3], "image_id": [], "input_seq": [None], 
                     "target_seq": [None], "indicator": [None], 
                     "image_features": [2048], "object_features": [num_boxes, 2048]}
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    x = iterator.get_next()
    image, image_id, input_seq, target_seq, indicator, image_features, object_features = (
        x["image"], x["image_id"], x["input_seq"], x["target_seq"], x["indicator"], 
        x["image_features"], x["object_features"])
    return image_id, image, image_features, object_features, input_seq, target_seq, indicator
    