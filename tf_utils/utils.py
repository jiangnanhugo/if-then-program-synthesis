import os
import errno
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layer_norm


def make_array(seqs, length=None, batch_size=None):
    '''Make a 2D NumPy array from a list of strings or a list of 1D arrays/lists.
    Shape of result is len(seqs) x length of longest sequence.'''
    if length is None:
        length = max(len(elem) for elem in seqs)
    if batch_size is None:
        array = np.full((len(seqs), length), 0, dtype=np.int32)
    else:
        array = np.full((batch_size, length), 0, dtype=np.int32)

    for i, item in enumerate(seqs):
        if isinstance(item, str):
            item = np.fromstring(item, np.uint8)
        array[i, :len(item)] = item

    return array

def create_buckets(sizes, min_bucket_size):
    '''Determine upper bounds for dividing |sizes| into buckets (contiguous
     ranges) of approximately equal size, where each bucket has at least
     |min_bucket_size|.'''

    sizes.sort()

    buckets = []
    bucket = []
    for size in sizes:
        if len(bucket) >= min_bucket_size and bucket[-1] < size:
            buckets.append(bucket[-1])
            bucket = []
        else:
            bucket.append(size)
    if bucket:
        buckets.append(bucket[-1])
    return buckets


def rnn(inputs, input_lengths, cell_type, num_layers, num_units, keep_prob, is_training,
        bid=False, residual=False, regular_output=False, use_norm=False):
    # inputs: batch x time x depth

    assert num_layers >= 1

    need_tuple_state = cell_type in (tf.nn.rnn_cell.BasicLSTMCell, tf.nn.rnn_cell.LSTMCell)

    if need_tuple_state:
        cell = cell_type(num_units, state_is_tuple=True)
    else:
        cell = cell_type(num_units)

    if is_training and keep_prob < 1:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-keep_prob)
    if residual:
        cell = tf.nn.rnn_cell.ResidualWrapper(cell)
    if bid:
        input_lengths_64 = tf.cast(input_lengths, tf.int64)
        prev_layer_fwd = inputs
        prev_layer_rev = tf.reverse_sequence(inputs, input_lengths_64, 1)
        for i in range(num_layers):
            with tf.variable_scope("Layer%d" % i):
                with tf.variable_scope("Fwd"):
                    outputs_fwd, final_state_fwd = tf.nn.dynamic_rnn(cell, prev_layer_fwd, input_lengths, dtype=tf.float32)
                    if use_norm:
                        outputs_fwd = layer_norm(outputs_fwd, scope="layer_norm")
                with tf.variable_scope("Rev"):
                    outputs_rev, final_state_rev = tf.nn.dynamic_rnn(cell, prev_layer_rev, input_lengths, dtype=tf.float32)
                    if use_norm:
                        outputs_rev = layer_norm(outputs_rev, scope="layer_norm")

                outputs_rev = tf.reverse_sequence(outputs_rev, input_lengths_64, 1)
                # print("{} {}".format(outputs_rev, outputs_fwd))
                prev_layer_fwd = tf.concat([outputs_fwd, outputs_rev], axis=2)
                prev_layer_rev = tf.reverse_sequence(prev_layer_fwd, input_lengths_64, 1)

        if regular_output:
            return prev_layer_fwd, final_state_fwd + final_state_rev

        if need_tuple_state:
            final_state_fwd = final_state_fwd[1]
            final_state_fwd.set_shape([inputs.get_shape()[0], cell.state_size[1]])
            final_state_rev = final_state_rev[1]
            final_state_rev.set_shape([inputs.get_shape()[0], cell.state_size[1]])
        else:
            final_state_fwd.set_shape([inputs.get_shape()[0], cell.state_size])
            final_state_rev.set_shape([inputs.get_shape()[0], cell.state_size])

        final_output = tf.concat([final_state_fwd, final_state_rev], axis=1)
        return prev_layer_fwd, final_output

    # Not bidirectional
    for i in range(num_layers):
        prev_layer = inputs
        with tf.variable_scope("Layer%d" % i):
            outputs, final_state = tf.nn.dynamic_rnn(cell, prev_layer, input_lengths, dtype=tf.float32)
            prev_layer = outputs

    if regular_output:
        return outputs, final_state

    if need_tuple_state:
        final_state[1].set_shape([inputs.get_shape()[0], cell.state_size[1]])
        return final_state[1]
    else:
        final_state.set_shape([inputs.get_shape()[0], cell.state_size])
        return final_state