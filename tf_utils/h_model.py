import collections
from .utils import rnn
from .h_softmax import *

label_names = ['trigger_chans', 'trigger_funcs', 'action_chans', 'action_funcs']


class IFTTTHModel(object):
    def __init__(self, config, vocab_size, memory_size, b_graph, is_train, stop_gradient):
        batch_size = config['batch_size']
        embed_size = config['embedding_size']
        decoder_type = config.get('decoder', 'standard')
        top_size = config['top_size']
        bottom_size = config['bottom_size']
        self.b_mapping = np.zeros((top_size, bottom_size), dtype=np.int32)
        self.mask = np.zeros((top_size, bottom_size), dtype=np.float32)
        for i, line in enumerate(b_graph):
            for j, x in enumerate(line):
                self.b_mapping[i, j] = x
            self.mask[i, :len(line)] = 1

        # placeholder
        self.ids = tf.placeholder(tf.int32, name='input', shape=[batch_size, memory_size])
        self.ids_lengths = tf.placeholder(tf.int32, name='input_lengths', shape=[batch_size])
        self.y_top = tf.placeholder(tf.int32, name='y_top', shape=[batch_size])
        self.y_bottom = tf.placeholder(tf.int32, name='y_bottom', shape=[batch_size])

        # batch size x seq length x label types
        ids = tf.minimum(vocab_size - 1, self.ids, name='make_unk')

        embedding = tf.get_variable('embed', [vocab_size, embed_size])
        input_x = tf.nn.embedding_lookup(embedding, ids)
        if config['cell_type'] == 'lstm':
            # outputs: RNN tensor list
            rnn_output, _ = rnn(input_x, self.ids_lengths,
                                tf.nn.rnn_cell.LSTMCell,
                                int(config['num_layers']),
                                int(config['num_units']),
                                config['keep_prob'],
                                is_train,
                                bid=config['bid'],
                                residual=True)
        elif config['cell_type'] == 'gru':
            # outputs: RNN tensor list
            rnn_output, _ = rnn(input_x, self.ids_lengths,
                                tf.nn.rnn_cell.GRUCell,
                                int(config['num_layers']),
                                int(config['num_units']),
                                config['keep_prob'],
                                is_train,
                                bid=config['bid'],
                                residual=True)
        elif config['name'] == 'Dict':
            rnn_output = input_x


        with tf.variable_scope('attention'):
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

        with tf.variable_scope('h_entropy'):
            self.top_W = tf.get_variable("top_weights", shape=[vec_size, top_size])
            self.top_b = tf.get_variable("top_bias",  shape=[top_size], initializer=tf.constant_initializer(0))
            self.bottom_W = tf.get_variable("bottom_weight", shape=[np.sum(self.mask), vec_size])
            # print(self.bottom_W)
            self.bottom_b = tf.get_variable("bottom_bias", shape=[bottom_size], initializer=tf.constant_initializer(0))
            target = (self.y_top, self.y_bottom)
            if stop_gradient:
                alpha = [0, 1]
            else:
                alpha = [1, 1]
            self.xentropy, self.bipart = h_softmax(logits, self.b_mapping, self.mask,                 # input
                                                       self.top_W, self.top_b, self.bottom_W, self.bottom_b,   # param
                                                       top_size, bottom_size, alpha, target)
            self.pred_y_top, self.pred_y_bottom = h_softmax(logits, self.b_mapping, self.mask,         # input
                                                                self.top_W, self.top_b, self.bottom_W, self.bottom_b,           # param
                                                                top_size, bottom_size, alpha)
            #     self.xentropy = l_softmax(logits, self.top_W, self.top_b,  top_size, target)
            #     self.pred_y_top = l_softmax(logits, self.top_W, self.top_b,  top_size)

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
                grads = tf.gradients(self.xentropy, tvars[-2:])#, stop_gradients=tvars[:-2])
            else:
                print()
                grads = tf.gradients(self.xentropy, tvars)
            if config['max_grad_norm'] > 0:
                grads, grads_norm = tf.clip_by_global_norm(grads, config['max_grad_norm'])
            else:
                grads_norm = tf.global_norm(grads)

            if stop_gradient:
                self.train_op = self.optimizer.apply_gradients(zip(grads, tvars[-2:]), global_step=self.global_step)
            else:
                self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)


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
