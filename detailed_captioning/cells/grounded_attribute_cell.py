'''Author: Brandon Trabucco, Copyright 2019
Implements a custom visual attention LSTM cell for image captioning.
Anderson, Peter, et al. https://arxiv.org/abs/1707.07998'''


import tensorflow as tf
import collections
from detailed_captioning.utils import tile_with_new_axis
from detailed_captioning.cells.attribute_caption_cell import AttributeCaptionCell


_GroundedAttributeStateTuple = collections.namedtuple("GroundedAttributeStateTuple", (
    "region", "attribute", "language"))
class GroundedAttributeStateTuple(_GroundedAttributeStateTuple):
    __slots__ = ()
    @property
    def dtype(self):
        (region, attribute, language) = self
        if region.dtype != attribute.dtype or region.dtype != language.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s %s" % (
                str(region.dtype), str(attribute.dtype), str(language.dtype)))
        return region.dtype
    
    
def softmax_attention(x):
    """Performs a softmax attention function across the second dim of the input.
    Args: x: a float32 Tensor with shape [batch_size, length, depth]
    Returns: a normalized Tensor with the same shape as x. """
    x = tf.transpose(x, [0, 2, 1])
    x = tf.nn.softmax(x)
    x = tf.transpose(x, [0, 2, 1])
    return x
    

class GroundedAttributeCell(AttributeCaptionCell):

    def __init__(self, 
            num_units, use_peepholes=False, cell_clip=None,
            initializer=None, num_proj=None, proj_clip=None,
            num_unit_shards=None, num_proj_shards=None,
            forget_bias=1.0, state_is_tuple=True,
            activation=None, reuse=None, name="grounded_attribute", dtype=None,
            mean_image_features=None, mean_object_features=None, 
            image_attribute_features=None, object_attribute_features=None, **kwargs ):
        super(GroundedAttributeCell, self).__init__(
            reuse=reuse, name=name, dtype=dtype,
            mean_image_features=mean_image_features, 
            mean_object_features=mean_object_features, 
            image_attribute_features=image_attribute_features,
            object_attribute_features=object_attribute_features, **kwargs )
        self.region_lstm = tf.contrib.rnn.LSTMCell(num_units, 
            use_peepholes=use_peepholes, cell_clip=cell_clip,
            initializer=initializer, num_proj=num_proj, proj_clip=proj_clip,
            num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
            forget_bias=forget_bias, state_is_tuple=state_is_tuple,
            activation=activation, reuse=reuse, name=(name + "/region"), dtype=dtype)
        self.region_sentinel_gate_layer = tf.layers.Dense(num_units, kernel_initializer=initializer, 
            name=(name + "/region_sentinel_gate_layer"), activation=tf.nn.sigmoid)
        self.region_sentinel_embeddings_layer = tf.layers.Dense(self.num_image_features, 
            kernel_initializer=initializer, name=(name + "/region_sentinel_embeddings_layer"), 
            activation=None)
        self.region_attention_layer = tf.layers.Dense(1, kernel_initializer=initializer, 
            name=(name + "/region_attention_layer"), activation=softmax_attention)
        self.attribute_lstm = tf.contrib.rnn.LSTMCell(num_units, 
            use_peepholes=use_peepholes, cell_clip=cell_clip,
            initializer=initializer, num_proj=num_proj, proj_clip=proj_clip,
            num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
            forget_bias=forget_bias, state_is_tuple=state_is_tuple,
            activation=activation, reuse=reuse, name=(name + "/attribute"), dtype=dtype)
        self.attribute_attention_layer = tf.layers.Dense(1, kernel_initializer=initializer, 
            name=(name + "/attribute_attention_layer"), activation=softmax_attention)
        self.attribute_sentinel_gate_layer = tf.layers.Dense(num_units, kernel_initializer=initializer, 
            name=(name + "/attribute_sentinel_gate_layer"), activation=tf.nn.sigmoid)
        self.attribute_sentinel_embeddings_layer = tf.layers.Dense(self.num_attribute_features, 
            kernel_initializer=initializer, name=(name + "/attribute_sentinel_embeddings_layer"), 
            activation=None)
        self.language_lstm = tf.contrib.rnn.LSTMCell(num_units, 
            use_peepholes=use_peepholes, cell_clip=cell_clip,
            initializer=initializer, num_proj=num_proj, proj_clip=proj_clip,
            num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
            forget_bias=forget_bias, state_is_tuple=state_is_tuple,
            activation=activation, reuse=reuse, name=(name + "/language"), dtype=dtype)
        self._state_size = GroundedAttributeStateTuple(self.region_lstm.state_size, 
            self.attribute_lstm.state_size, self.language_lstm.state_size)
        self._output_size = self.language_lstm.output_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state):
        region_inputs = tf.concat([tf.concat(state.language, 1), tf.concat(state.attribute, 1), 
            self.mean_image_features, inputs], 1)
        region_outputs, region_next_state = self.region_lstm(region_inputs, state.region)
        region_sentinel = self.region_sentinel_embeddings_layer(tf.nn.tanh(
            region_next_state.c) * self.region_sentinel_gate_layer(tf.concat([
            state.region.h, inputs], 1)))
        region_features = tf.concat([self.mean_object_features, tf.expand_dims(
            self.mean_image_features, 1), tf.expand_dims(region_sentinel, 1)], 1)
        region_attention_inputs = tf.concat([region_features, tile_with_new_axis(
            region_outputs, [tf.shape(region_features)[1]], [1])], 2)
        region_attention_mask = self.region_attention_layer(region_attention_inputs)
        region_pointer_ids = tf.argmax(tf.squeeze(region_attention_mask, 2), 1, output_type=tf.int32)
        attended_region_features = tf.reduce_sum(region_features * region_attention_mask, 1)
        attribute_inputs = tf.concat([region_outputs, tf.concat(state.language, 1), 
            attended_region_features], 1)
        attribute_outputs, attribute_next_state = self.attribute_lstm(attribute_inputs, state.attribute)
        attribute_sentinel = self.attribute_sentinel_embeddings_layer(tf.nn.tanh(
            attribute_next_state.c) * self.attribute_sentinel_gate_layer(tf.concat([
            state.attribute.h, inputs], 1)))
        all_attribute_features = tf.concat([
            self.object_attribute_features, 
            tf.expand_dims(self.image_attribute_features, 1),
            tf.expand_dims(tile_with_new_axis(attribute_sentinel, [tf.shape(
                self.image_attribute_features)[1]], [1]), 1)], 1)
        attribute_features = tf.gather_nd(all_attribute_features, 
            tf.concat([tf.expand_dims(tf.range(tf.shape(region_pointer_ids)[0]), 1), 
                       tf.expand_dims(region_pointer_ids, 1)], 1))
        attribute_attention_inputs = tf.concat([attribute_features, tile_with_new_axis(
            attribute_outputs, [tf.shape(attribute_features)[1]], [1])], 2)
        attribute_attention_mask = self.attribute_attention_layer(attribute_attention_inputs)
        attended_attribute_features = tf.reduce_sum(attribute_features * attribute_attention_mask, 1)
        language_inputs = tf.concat([attribute_outputs, attended_attribute_features], 1)
        language_outputs, language_next_state = self.language_lstm(language_inputs, state.language)
        return language_outputs, GroundedAttributeStateTuple(region_next_state, 
            attribute_next_state, language_next_state)
    
    @property
    def trainable_variables(self):
        cell_variables = ( self.region_lstm.trainable_variables 
            + self.region_sentinel_gate_layer.trainable_variables
            + self.region_sentinel_embeddings_layer.trainable_variables
            + self.region_attention_layer.trainable_variables
            + self.attribute_lstm.trainable_variables 
            + self.attribute_sentinel_gate_layer.trainable_variables
            + self.attribute_sentinel_embeddings_layer.trainable_variables
            + self.attribute_attention_layer.trainable_variables
            + self.language_lstm.trainable_variables )
        return cell_variables
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        cell_variables = ( self.region_lstm.variables 
            + self.region_sentinel_gate_layer.variables
            + self.region_sentinel_embeddings_layer.variables
            + self.region_attention_layer.variables
            + self.attribute_lstm.variables 
            + self.attribute_sentinel_gate_layer.variables
            + self.attribute_sentinel_embeddings_layer.variables
            + self.attribute_attention_layer.variables
            + self.language_lstm.variables )
        return cell_variables
    
    @property
    def weights(self):
        return self.variables
    
