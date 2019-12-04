from .utils import rnn
import tensorflow as tf
import numpy as np


class IFTTTHHModel(object):
    def __init__(self, config, vocab_size, memory_size, maps, is_train, stop_gradient):
        batch_size = config['batch_size']
        embed_size = config['embedding_size']
        decoder_type = config.get('decoder', 'standard')
        bmap_shape = []
        b_mappings = []
        masks = []
        for map in maps:
            bmap_shape.append(map['shape'])
            b_mapping = np.zeros(map['shape'][:2], dtype=np.int32)
            mask = np.zeros(map['shape'][:2], dtype=np.float32)
            for i, line in enumerate(map['mapping']):
                for j, x in enumerate(line):
                    b_mapping[i, j] = x
                mask[i, :len(line)] = 1
            b_mappings.append(b_mapping)
            masks.append(mask)
        self.bmap = b_mappings
        # placeholder
        self.ids = tf.placeholder(tf.int32, name='input', shape=[batch_size, memory_size])
        self.ids_lengths = tf.placeholder(tf.int32, name='input_lengths', shape=[batch_size])
        self.y_top = tf.placeholder(tf.int32, name='y', shape=[batch_size, 2])
        self.y_bottom = tf.placeholder(tf.int32, name='y_bottom', shape=[batch_size, 2])

        # batch size x seq length x label types
        ids = tf.minimum(vocab_size - 1, self.ids, name='make_unk')

        self.loss = 0
        pred_top, pred_bottom = [], []
        with tf.variable_scope('text_vector'):
            embedding = tf.get_variable('embed', [vocab_size, embed_size])
            input_x = tf.nn.embedding_lookup(embedding, ids)
            if config['cell_type'] == 'lstm':
                # outputs: RNN tensor list
                rnn_output, _ = rnn(input_x, self.ids_lengths, tf.nn.rnn_cell.LSTMCell,
                                    int(config['num_layers']), int(config['num_units']), config['keep_prob'],
                                    is_train, bid=config['bid'], residual=True, use_norm=config['use_layer_norm'])
            elif config['cell_type'] == 'gru':
                # outputs: RNN tensor list
                rnn_output, _ = rnn(input_x, self.ids_lengths, tf.nn.rnn_cell.GRUCell,
                                    int(config['num_layers']), int(config['num_units']), config['keep_prob'],
                                    is_train, bid=config['bid'], residual=True, use_norm=config['use_layer_norm'])
            elif config['name'] == 'Dict':
                rnn_output = input_x
            # rnn_outputs.append(rnn_output)
            with tf.variable_scope('attention_'+str(i)):
                if decoder_type == 'attention':
                    if config['cell_type'] == 'lstm' or config['cell_type'] == 'gru':
                        vec_size = embed_size * 2
                    else:
                        vec_size = embed_size
                    TA = tf.get_variable("BIAS_VECTOR", [memory_size, vec_size])
                    m = rnn_output + TA
                    PREP = tf.get_variable("PREP", [1, vec_size])
                    dotted_prep = tf.reduce_sum(m * PREP, 1)
                    probs_prep = tf.nn.softmax(dotted_prep)
                    probs_prep_temp = tf.transpose(tf.expand_dims(probs_prep, -1), [0, 2, 1])
                    dotted = tf.reduce_sum(m * probs_prep_temp, 2)
                    output_probs = tf.nn.softmax(dotted)
                    probs_temp = tf.expand_dims(output_probs, 1)

                    c_temp = tf.transpose(m, [0, 2, 1])
                    logits = tf.reduce_sum(c_temp * probs_temp, 2)

        for i in range(2):
                with tf.variable_scope('hh_entropy'+str(i)):
                    # initizer:tf.contrib.layers.xavier_initializer()
                    top_W = tf.get_variable("top_weights", shape=[vec_size, bmap_shape[i][0]], initializer=tf.contrib.layers.variance_scaling_initializer())
                    top_b = tf.get_variable("top_bias",  shape=[bmap_shape[i][0]], initializer=tf.constant_initializer(0))
                    bottom_W = tf.get_variable("bottom_weight", shape=[bmap_shape[i][2], vec_size], initializer=tf.contrib.layers.variance_scaling_initializer())
                    bottom_b = tf.get_variable("bottom_bias", shape=[bmap_shape[i][1]], initializer=tf.constant_initializer(0))
                    target = (self.y_top[:, i], self.y_bottom[:, i])
                    if stop_gradient:
                        alpha = [0, 1]
                    else:
                        alpha = [1, 1]
                    print(bmap_shape[i])
                    xentropy, self.bipt = h_softmax(logits, b_mappings[i], masks[i],                 # input
                                            top_W, top_b, bottom_W, bottom_b,   # param
                                            bmap_shape[i][0], bmap_shape[i][1], target)
                    pred_y_top, pred_y_bottom = h_softmax(logits, b_mappings[i], masks[i],         # input
                                                          top_W, top_b, bottom_W, bottom_b,           # param
                                                          bmap_shape[i][0], bmap_shape[i][1])
                    self.loss += xentropy
                    pred_top.append(tf.expand_dims(pred_y_top, axis=1))
                    pred_bottom.append(tf.expand_dims(pred_y_bottom, axis=1))
        self.pred_top = tf.concat(pred_top, axis=1)
        self.pred_bottom = tf.concat(pred_bottom, axis=1)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        if is_train:
            lr = tf.train.exponential_decay(config['lr_scheduler']['init_lr'],
                                            global_step=self.global_step,
                                            decay_rate=config['lr_scheduler']['decay_rate'],
                                            decay_steps=config['lr_scheduler']['decay_steps'])

            tvars = tf.trainable_variables()
            if config['optimizer'] == "AdamOptimizer":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            else:
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)

            if stop_gradient:
                grads = tf.gradients(self.loss, tvars[-2:])#, stop_gradients=tvars[:-2])
            else:
                grads = tf.gradients(self.loss, tvars)
            if config['max_grad_norm'] > 0:
                grads, grads_norm = tf.clip_by_global_norm(grads, config['max_grad_norm'])
            else:
                grads_norm = tf.global_norm(grads)

            if stop_gradient:
                self.train_op = self.optimizer.apply_gradients(zip(grads, tvars[-2:]), global_step=self.global_step)
            else:
                self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)


def h_softmax(hidden_state, b_mapping, b_mask,             # input
              top_W, top_b, bottom_W, bottom_b,            # param
              top_size, bottom_size,target=None):         # shape
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
        log_loss = tf.reduce_mean(target_class_loss + target_output_loss, axis=1, keep_dims=True)

        return -log_loss, batch_bipartite