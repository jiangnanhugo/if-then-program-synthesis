from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import numpy as np


def h_softmax(hidden_state, b_mapping, b_mask,             # input
              top_W, top_b, bottom_W, bottom_b,            # param
              top_size, bottom_size, alpha, target=None):         # shape
    # First softmax that computes the probabilities of belonging to each class
    class_probs = tf.nn.log_softmax(tf.matmul(hidden_state, top_W) + top_b)  # [batch_size, n_classes]
    if target == None:
        y_top = tf.argmax(class_probs, axis=1)
        batch_bipartite = tf.gather(b_mapping, y_top)
        batch_mask = tf.gather(b_mask, y_top)
        _W2 = tf.gather(bottom_W, batch_bipartite)                     # [batch_size, n_outputs_per_class]
        _b2 = tf.gather(bottom_b, batch_bipartite)                     # [batch_size]
        activations = tf.matmul(_W2, tf.expand_dims(hidden_state, 2))  # [batch_size, n_outputs_per_class]
        activations = tf.squeeze(activations) + _b2                    # [batch_size, n_outputs_per_class]
        output_probs = tf.nn.log_softmax(activations * batch_mask)
        y_bottom = tf.argmax(output_probs, axis=1)
        return y_top, y_bottom
    else:
        y_top, y_bottom = target
        batch_bipartite = tf.gather(b_mapping, y_top)
        batch_mask = tf.gather(b_mask, y_top)
        _W2 = tf.gather(bottom_W, batch_bipartite)  # [batch_size, ]
        _b2 = tf.gather(bottom_b, batch_bipartite)  # [batch_size, ]
        activations = tf.matmul(_W2, tf.expand_dims(hidden_state, 2))  # [batch_size, n_outputs_per_class]
        activations = tf.squeeze(activations) + _b2  # [batch_size, n_outputs_per_class]
        output_probs = tf.nn.log_softmax(activations * batch_mask)
        target_class_loss = tf.reduce_sum(class_probs * tf.one_hot(y_top, depth=top_size),
                                          axis=1,
                                          keep_dims=True)  # [batch_size, 1] for probs be in the target_classes

        target_output_loss = tf.reduce_sum(output_probs * tf.one_hot(y_bottom, depth=bottom_size),
                                           axis=1,
                                           keep_dims=True)  # [batch_size, 1] for probs be the target_outputs_in_class
        log_loss = tf.reduce_mean(alpha[0]*target_class_loss + alpha[1]*target_output_loss)

        return -log_loss, batch_bipartite

def test_h_softmax():
    batch_size = 29
    vec_size = 30
    # len of services: 2885, len of functions: 6932
    top_size = 2885
    bottom_size = 32
    y_top = np.random.randint(low=0, high=top_size, size=batch_size)
    y_bottom = np.random.randint(low=0, high=9, size=batch_size)
    b_mask = np.random.randint(low=0, high=2, size=(batch_size, 9))

    top_bottom_bipt = np.random.randint(low=1, high=bottom_size, size=(top_size, 9)).astype(np.int32)

    hidden_state = np.random.rand(batch_size, vec_size).astype(np.float32)

    with tf.variable_scope('entropy'):
        top_W = tf.get_variable("top_weights", shape=[vec_size, top_size])
        top_b = tf.get_variable("top_bias", shape=[top_size], initializer=tf.constant_initializer(0))
        bottom_W = tf.get_variable("bottom_weight", shape=[bottom_size, vec_size])
        bottom_b = tf.get_variable("bottom_bias", shape=[bottom_size], initializer=tf.constant_initializer(0))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    class_probs = tf.nn.softmax(tf.matmul(hidden_state, top_W) + top_b)
    target_class_probs = tf.reduce_sum(class_probs * tf.one_hot(y_top, depth=top_size),
                                       axis=1,
                                       keep_dims=True)  # [batch_size, 1] for probs be in the target_classes
    batch_bipartite = tf.gather(top_bottom_bipt, y_top)
    _W2 = tf.gather(bottom_W, batch_bipartite)  # [batch_size, ]
    _b2 = tf.gather(bottom_b, batch_bipartite)
    activations = tf.matmul(_W2, tf.expand_dims(hidden_state, 2))
    activations = tf.squeeze(activations) + _b2
    output_probs = tf.nn.softmax(activations*b_mask)
    target_output_probs = tf.reduce_sum(output_probs * tf.one_hot(y_bottom, depth=9),
                                        axis=1, keep_dims=True)  # [batch_size, 1] for probs be the target_outputs_in_class
    probs = target_class_probs * target_output_probs
    return probs


def l_softmax(hidden_state, top_W, top_b, top_size, target=None):         # shape

    # First softmax that computes the probabilities of belonging to each class
    class_probs = tf.nn.log_softmax(tf.matmul(hidden_state, top_W) + top_b)  # [batch_size, n_classes]
    if target == None:
        y_top = tf.argmax(class_probs, axis=1)
        return y_top
    else:
        y_top, y_bottom = target
        target_class_loss = tf.reduce_sum(class_probs * tf.one_hot(y_top, depth=top_size),
                                          axis=1,
                                          keep_dims=True)  # [batch_size, 1] for probs be in the target_classes
        return -target_class_loss


