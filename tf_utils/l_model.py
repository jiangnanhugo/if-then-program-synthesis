import collections
from .utils import rnn
import tensorflow as tf

label_names = ['trigger_chans', 'trigger_funcs', 'action_chans', 'action_funcs']


class IFTTTLModel(object):
    def __init__(self, config, vocab_size, memory_size, label_size, is_train):
        batch_size = config['batch_size']
        embed_size = config['embedding_size']
        decoder_type = config.get('decoder', 'standard')
        # placeholder
        self.ids = tf.placeholder(tf.int32, name='input', shape=[batch_size, memory_size])
        self.ids_lengths = tf.placeholder(tf.int32, name='input_lengths', shape=[batch_size])
        self.label = tf.placeholder(tf.int32, name='label', shape=[batch_size, 4])

        # batch size x seq length x label types
        ids = tf.minimum(vocab_size - 1, self.ids, name='make_unk')

        embed_x = tf.get_variable('embed_x', [vocab_size, embed_size])
        input_x = tf.nn.embedding_lookup(embed_x, ids)
        rnn_outputs = []
        for i in range(4):
            with tf.variable_scope('lstm_for_'+str(i)):
                if config['cell_type'] == 'lstm':
                    # outputs: RNN tensor list
                    rnn_output, _ = rnn(input_x, self.ids_lengths,
                                        tf.nn.rnn_cell.GRUCell,
                                        int(config['num_layers']),
                                        int(config['num_units']),
                                        config['keep_prob'],
                                        is_train,
                                        bid=config['bidirectional'])
                    rnn_outputs.append(rnn_output)
                elif config['name'] == 'Dict':
                    rnn_output = input_x
                    rnn_outputs.append(rnn_output)

        pred = []
        loss = 0
        for i in range(4):
            with tf.variable_scope('attention' + str(i)):
                if decoder_type == 'attention':
                    if config['cell_type'] == 'lstm':
                        vec_size = embed_size * 2
                    else:
                        vec_size = embed_size
                    TA = tf.get_variable("BIAS_VECTOR", [memory_size, vec_size])
                    m = rnn_outputs[i] + TA
                    PREP = tf.get_variable("PREP", [1, vec_size])
                    dotted_prep = tf.reduce_sum(m * PREP, 1)
                    probs_prep = tf.nn.softmax(dotted_prep)
                    probs_prep_temp = tf.transpose(tf.expand_dims(probs_prep, -1), [0, 2, 1])
                    dotted = tf.reduce_sum(m * probs_prep_temp, 2)
                    output_probs = tf.nn.softmax(dotted)
                    probs_temp = tf.expand_dims(output_probs, 1)

                    c_temp = tf.transpose(m, [0, 2, 1])
                    o_k = tf.reduce_sum(c_temp * probs_temp, 2)
                    softmax_w = tf.get_variable("softmax_w", [vec_size, label_size[i]], initializer=tf.contrib.layers.xavier_initializer())
                    # print('o_k:',o_k)
                    # print("softmax_w",softmax_w)
                    logits = tf.matmul(o_k, softmax_w)

            with tf.variable_scope('entropy' + str(i)):
                log_probs = tf.nn.log_softmax(logits)
                pred.append(tf.argmax(log_probs, axis=1))
                loss += tf.reduce_sum(log_probs * tf.one_hot(self.label[:, i], depth=label_size[i]),
                                      axis=1,
                                      keep_dims=True)
        self.xentropy = -loss
        self.pred = tf.transpose(tf.stack(pred))

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        if is_train:
            self.train_op = make_train_op(self.global_step, self.xentropy, config)


class keydefaultdict(collections.defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def without_name(d):
    d = d.copy()
    del d['name']
    return d


def make_train_op(global_step, loss, config):
    lr = tf.train.exponential_decay(config['lr_scheduler']['init_lr'],
                                    global_step=global_step,
                                    decay_rate=0.9,
                                    decay_steps=1000)

    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    if config['max_grad_norm'] > 0:
        grads, grads_norm = tf.clip_by_global_norm(grads, config['max_grad_norm'])
    else:
        grads_norm = tf.global_norm(grads)
    if config['optimizer'] == "AdamOptimizer":
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    else:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)

    return optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

