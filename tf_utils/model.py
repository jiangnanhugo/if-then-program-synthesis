import collections
import tensorflow as tf
from .utils import rnn


label_names = ['trigger_chans', 'trigger_funcs', 'action_chans', 'action_funcs']


class IFTTTModel(object):
    def __init__(self, optim_config, model_config, arch_config, num_labels,
                 label_types, vocab_size, memory_size, is_training):
        batch_size = optim_config['batch_size']
        self.ids = tf.placeholder(tf.int32, name='input', shape=[batch_size, memory_size])
        self.ids_lengths = tf.placeholder(tf.int32, name='input_lengths',  shape=[batch_size])
        self.labels = tf.placeholder(tf.int32, name='labels', shape=[batch_size, len(num_labels)])
        # batch size x seq length x label types

        ids = tf.minimum(vocab_size - 1, self.ids, name='make_unk')
        if arch_config:
            # label_groupings=[[0],[1],[2],[3],[4]]
            label_groupings = arch_config['label_groupings']

            if arch_config['share_word_embeddings']:
                # print("use shared word embeddings")
                embedding = tf.get_variable('shared_embed', [vocab_size, model_config['embedding_size']])
                rnn_inputs = [tf.nn.embedding_lookup(embedding, ids)] * len(label_groupings)
                # print(rnn_inputs)
            else:
                # print('NO use of shared word embedding')
                rnn_inputs = []
                for grouping in label_groupings:
                    embedding = tf.get_variable('embed_' + '_'.join(str(i) for i in grouping),
                                                [vocab_size, model_config['embedding_size']])
                    rnn_inputs.append(tf.nn.embedding_lookup(embedding, ids))

            outputs = []
            for grouping, rnn_input in zip(label_groupings, rnn_inputs):
                with tf.variable_scope('labels_' + '_'.join(str(i) for i in grouping)):
                    if model_config['name'] == 'rnn':
                        outputs.append(rnn(rnn_input,
                                           self.ids_lengths,
                                           getattr(tf.nn.rnn_cell, model_config['cell_type']),
                                           int(model_config['num_layers']),
                                           int(model_config['num_units']),
                                           model_config['keep_prob'],
                                           is_training,
                                           bid=model_config['bidirectional']))
                    elif model_config['name'] == 'Dict':
                        outputs.append((rnn_input, tf.reduce_sum(rnn_input, 1)))

        else:
            label_groupings = [label_types]
            embedding = tf.get_variable('embedding', [vocab_size, model_config['embedding_size']])
            rnn_input = tf.nn.embedding_lookup(embedding, ids)
            if model_config['name'] == 'rnn':
                # outputs: RNN tensor list
                outputs = [rnn(rnn_input,
                               self.ids_lengths,
                               getattr(tf.nn.rnn_cell, model_config['cell_type']),
                               int(model_config['num_layers']),
                               int(model_config['num_units']),
                               model_config['keep_prob'],
                               is_training,
                               bid=model_config['bidirectional'],
                               residual=False)]
            else:
                outputs = [(rnn_input, tf.reduce_sum(rnn_input, 1))]

        losses = []
        all_preds = []
        self.all_probs = []
        self.softmax_placehold = []
        self.softmax_init = []
        decoder_type = model_config.get('decoder', 'standard')

        # print('num_labels:', num_labels)
        # print('label_types:', label_types)
        # print('outputs:', outputs)
        # print('label_groups:', label_groupings)
        for i, num_classes in enumerate(num_labels):
            with tf.variable_scope('label_{}'.format(label_types[i])):
                if model_config['name'].find('rnn') != -1:
                    vec_size = model_config['embedding_size'] * 2
                else:
                    vec_size = model_config['embedding_size']

                TA = tf.get_variable("BIAS_VECTOR", [memory_size, vec_size])
                # print("the trick TA: {}".format(TA))
                m = outputs[i][0] + TA # states + TA

                PREP = tf.get_variable("PREP", [1, vec_size])
                # print("{}-th num_classes: {}".format(i, num_classes))

                if decoder_type == 'LA':
                    B = tf.get_variable("B", [vec_size, memory_size])
                    m_t = tf.reshape(m, [batch_size * memory_size, vec_size])
                    d_t = tf.matmul(m_t, B)
                    d_softmax = tf.nn.softmax(d_t)
                    d = tf.reshape(d_softmax, [batch_size, memory_size, memory_size])
                    dotted_prep = tf.reduce_sum(outputs[i][0] * PREP, 2)
                elif decoder_type == 'attention':
                    dotted_prep = tf.reduce_sum(m * PREP, 1)

                probs_prep = tf.nn.softmax(dotted_prep)
                preps = []
                preps.append(probs_prep)
                for _ in range(1):
                    probs_prep = preps[-1]
                    if decoder_type == 'LA':
                        probs_prep_temp = tf.expand_dims(probs_prep, -1)
                        # batch_matmul -> matmul
                        probs_temp = tf.matmul(d, probs_prep_temp)
                        probs = tf.squeeze(probs_temp)
                        output_probs = tf.nn.l2_normalize(probs, 1)
                    elif decoder_type == 'attention':
                        probs_prep_temp = tf.transpose(tf.expand_dims(probs_prep, -1), [0, 2, 1])
                        dotted = tf.reduce_sum(m * probs_prep_temp, 2)
                        output_probs = tf.nn.softmax(dotted)
                    preps.append(output_probs)

                probs_temp = tf.expand_dims(preps[-1], 1)
                c_temp = tf.transpose(m, [0, 2, 1])
                mat_temp = c_temp * probs_temp
                o_k = tf.reduce_sum(mat_temp, 2)
                # softmax_w = tf.Variable(tf.constant(0.0, shape=[vec_size, num_classes]), trainable=True, name="softmax_w")
                softmax_w = tf.get_variable("softmax_w", [vec_size, num_classes])
                # softmax_w_placehold = tf.placeholder(tf.float32, [vec_size, num_classes])
                # self.softmax_init.append(softmax_w.assign(softmax_w_placehold))
                self.softmax_placehold.append(softmax_w)
                logits = tf.matmul(o_k, softmax_w)

                preds = tf.cast(tf.argmax(logits, 1), tf.int32)
                all_preds.append(preds)
                probs = tf.nn.softmax(logits)
                self.all_probs.append(probs)
                xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels[:, i],
                                                                          logits=logits,
                                                                          name='loss_function')
                loss = tf.reduce_mean(xentropy)
                losses.append(loss)

        self.preds = tf.transpose(tf.stack(all_preds))
        self.loss = tf.add_n(losses)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        if is_training:
            self.train_op = make_train_op(self.global_step, self.loss, optim_config)


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


def make_train_op(global_step, loss, optim_config):
    lr = getattr(tf.train,
                 optim_config['lr_scheduler']['name'])(global_step=global_step,
                                                       **without_name(optim_config['lr_scheduler']))

    tvars = tf.trainable_variables()
    # for item in tvars:
    #     print(item)
    grads = tf.gradients(loss, tvars)
    if optim_config['max_grad_norm'] > 0:
        grads, grads_norm = tf.clip_by_global_norm(grads, optim_config['max_grad_norm'])
    else:
        grads_norm = tf.global_norm(grads)
    # tf.summary.scalar('grads/norm', grads_norm)
    if optim_config['optimizer']['name'] == "AdamOptimizer":
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    else:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)

    return optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
