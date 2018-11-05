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
            "image/scores": tf.FixedLenSequenceFeature([], dtype=tf.float32),
            "image/scores_shape": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "image/boxes": tf.FixedLenSequenceFeature([], dtype=tf.float32),
            "image/boxes_shape": tf.FixedLenSequenceFeature([], dtype=tf.int64)})
    image, image_id, caption = (
        context["image/data"], context["image/image_id"], sequence["image/caption_ids"])
    scores = tf.reshape(sequence["image/scores"], sequence["image/scores_shape"])
    boxes = tf.reshape(sequence["image/boxes"], sequence["image/boxes_shape"])
    return {"image": image, "image_id": image_id, "caption": caption, 
            "scores": scores, "boxes": boxes}


def _decode_and_resize_image(x):
    image, image_id, caption, scores, boxes = (
        x["image"], x["image_id"], x["caption"], x["scores"], x["boxes"])
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, size=[800, 800], 
                                   method=tf.image.ResizeMethod.BILINEAR)
    return {"image": image, "image_id": image_id, "caption": caption, 
            "scores": scores, "boxes": boxes}
    

def _random_distort(x):
    image, image_id, caption, scores, boxes = (
        x["image"], x["image_id"], x["caption"], x["scores"], x["boxes"])
    image = tf.random_crop(image, [640, 640, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.032)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return {"image": image, "image_id": image_id, "caption": caption, 
            "scores": scores, "boxes": boxes}


def _crop_or_pad(x):
    image, image_id, caption, scores, boxes = (
        x["image"], x["image_id"], x["caption"], x["scores"], x["boxes"])
    image = tf.image.resize_image_with_crop_or_pad(image, 640, 640)
    return {"image": image, "image_id": image_id, "caption": caption, 
            "scores": scores, "boxes": boxes}


def _mask_and_slice(x):
    image, image_id, caption, scores, boxes = (
        x["image"], x["image_id"], x["caption"], x["scores"], x["boxes"])
    image = tf.multiply(tf.subtract(image, 0.5), 2.0)
    caption_length = tf.shape(caption)[0]
    input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)
    input_seq = tf.slice(caption, [0], input_length)
    target_seq = tf.slice(caption, [1], input_length)
    indicator = tf.ones(input_length, dtype=tf.int32)
    return {"image": image, "image_id": image_id, "input_seq": input_seq, 
            "target_seq": target_seq, "indicator": indicator, 
            "scores": scores, "boxes": boxes}


def _convert_dtype(x):
    image, image_id, input_seq, target_seq, indicator, scores, boxes = (
        x["image"], x["image_id"], x["input_seq"], x["target_seq"], x["indicator"], 
        x["scores"], x["boxes"])
    image = tf.cast(image, tf.float32)
    image_id = tf.cast(image_id, tf.int32)
    input_seq = tf.cast(input_seq, tf.int32)
    target_seq = tf.cast(target_seq, tf.int32)
    indicator = tf.cast(indicator, tf.float32)
    return {"image": image, "image_id": image_id, "input_seq": input_seq, 
            "target_seq": target_seq, "indicator": indicator, 
            "scores": scores, "boxes": boxes}


def _load_dataset_from_tf_records(is_training):
    if is_training:
        input_file_pattern = "data/coco/train-?????-of-?????"
    if not is_training:
        input_file_pattern = "data/coco/val-?????-of-?????"
    data_files = []
    for pattern in input_file_pattern.split(","):
        data_files.extend(tf.gfile.Glob(pattern))
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


def import_mscoco(is_training=True, batch_size=32, num_epochs=1, k=8):
    dataset = _load_dataset_from_tf_records(is_training)
    dataset = _apply_dataset_transformations(dataset, is_training)
    dataset = dataset.shuffle(buffer_size=1000)
    padded_shapes = {"image": [640, 640, 3], "image_id": [], "input_seq": [None], 
                     "target_seq": [None], "indicator": [None], 
                     "scores": [100], "boxes": [100, 4]}
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    x = iterator.get_next()
    image, image_id, input_seq, target_seq, indicator, scores, boxes = (
        x["image"], x["image_id"], x["input_seq"], x["target_seq"], x["indicator"], 
        x["scores"], x["boxes"])
    scores, top_k = tf.nn.top_k(scores, k=k)
    batch_ids = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, k])
    boxes = tf.gather_nd(boxes, tf.stack([batch_ids, top_k], 2))
    return image_id, image, scores, boxes, input_seq, target_seq, indicator
    