import argparse
import bisect
import pickle
from collections import defaultdict
import json
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
import sys
import datetime
import tempfile
from tqdm import tqdm
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import tf_utils
from tf_utils import IFTTTLModel
import numpy as np
np.set_printoptions(precision=4)


class MainLoop(object):
    def __init__(self, args, config):
        self.config = config
        self.root_logdir = args.logdir
        self.output_file = args.output
        self.label_size = [112, 495, 87, 201]
        self.data = pickle.load(open(args.dataset, 'rb'), encoding="latin1")
        self.max_desc_length = 0

        for item in (self.data['train'] + self.data['dev']):
            self.max_desc_length = max(self.max_desc_length, len(item['ids']))

        self.memory_size = self.config['memory_size']

        for section in ('train', 'test', 'dev'):
            for i in range(len(self.data[section])):
                if len(self.data[section][i]['ids']) > self.memory_size:
                    d = self.data[section][i]['ids']
                    d = d[:(self.memory_size + 1) // 2] + d[len(d) - self.memory_size//2:]
                    self.data[section][i]['ids'] = d

        buckets = tf_utils.create_buckets([len(item['ids']) for item in self.data['train']],
                                          self.config['batch_size'] * 5000)
        print('The bucket size:', buckets)

        self.bucketed_train = [[]] * len(buckets)
        for item in self.data['train']:
            size = len(item['ids'])
            self.bucketed_train[bisect.bisect_left(buckets, size)].append(item)

        bucketed_train_lens = np.array([len(bucket) for bucket in self.bucketed_train], dtype=float)
        self.bucket_dist = bucketed_train_lens / np.sum(bucketed_train_lens)

        scale = self.config['init']['scale']
        self.initializer = getattr(tf, self.config['init']['name'])(-scale, scale)

        max_word_id = int(self.config['max_word_id'])
        if max_word_id > 0:
            self.vocab_size = max_word_id
        else:
            self.vocab_size = len(self.data['word_ids'])

    def read(self, cur_point):
        if cur_point == 0:
            random.shuffle(self.bucketed_train[0])

        bucket = self.bucketed_train[0]
        batch = bucket[cur_point:cur_point + self.config['batch_size']]
        batch = batch + [random.choice(bucket)
                         for _ in range(self.config['batch_size'] - len(batch))]
        return self.get_batch(batch)

    def get_batch(self, batch):
        # [batch_size(32), max_seq_len(25)]
        ids = tf_utils.make_array([item['ids'] for item in batch], length=self.memory_size)
        # [batch_size(32)]
        ids_lengths = np.array([len(item['ids']) for item in batch])
        # [batch_size(32), label types(4)]
        labels = np.asarray([x['labels'] for x in batch], dtype=np.int32)
        return ids, ids_lengths, labels

    def get_batch_full(self, batch, batch_size):
        ids = tf_utils.make_array([item['ids'] for item in batch], length=self.memory_size, batch_size=batch_size)
        ids_lengths = np.zeros(batch_size)
        for i, item in enumerate(batch):
            ids_lengths[i] = len(item['ids'])
        labels = np.zeros((batch_size, 4))
        for i, item in enumerate(batch):
            labels[i] = item['labels']

        return ids, ids_lengths, labels

    def create_model(self, is_train):
        return IFTTTLModel(self.config, self.vocab_size, self.memory_size, self.label_size, is_train)

    def run(self, initializer, logdir, load_model=None):
        epoches = self.config['epoches']
        batch_size = self.config['batch_size']
        with tf.variable_scope('model', reuse=None, initializer=initializer):
            model = self.create_model(is_train=True)
        with tf.variable_scope('model', reuse=True, initializer=initializer):
            model_valid = self.create_model(is_train=False)
        saver = tf.train.Saver(max_to_keep=0)
        session = tf.Session()
        # intialize the params
        if load_model:
            saver.restore(session, os.path.join(args.logdir, 'model.ckpt-{}'.format(load_model)))
            self.generate_test_summaries(session, model)
            return
            initial_global_step = model.global_step.assign(50)
            session.run(initial_global_step)
        else:
            session.run(tf.global_variables_initializer())
        # Disable automatic saving and deleting
        supervisor = tf.train.Supervisor(logdir=logdir, summary_op=None, save_model_secs=0, saver=saver)

        epoch = 0
        cur_point = 0
        best_acc = 0.
        while not supervisor.should_stop() and epoch <= epoches:
            epoch += 1
            iterations = int(len(self.bucketed_train[0]) / batch_size)
            c1 = 0
            loss = 0.
            for _ in tqdm(range(iterations), leave=False):
                batch = self.read(cur_point)
                cur_point += len(batch[0])
                # Run 1 step with mini-batch
                input_list = [model.xentropy, model.pred, model.train_op, model.global_step]
                ids, ids_lengths, label = batch
                feed_dict = dict({model.ids: ids,
                                  model.ids_lengths: ids_lengths,
                                  model.label: label})
                cost, pred, _, global_step = session.run(input_list, feed_dict)
                loss += np.sum(cost)
                correct = (pred[:batch_size] == label[:batch_size])
                c1 += np.sum(correct)
                # break
            # reset data pointer
            cur_point = 0
            # print("c1:", format(c1/iterations, '.2f'), "loss:", format(loss/iterations, '.2f'), end="  ")

            cur_acc = self.generate_test_summaries(session, model_valid)

            if cur_acc > best_acc:
                best_acc = cur_acc
                print("save acc:", format(cur_acc, '.4f'), 'time:', datetime.datetime.now(), end=' ')
            print()
            # supervisor.saver.save(session, supervisor.save_path, global_step=global_step)

    def generate_test_summaries(self, session, model):
        batch_size = self.config['batch_size']
        num_correct = defaultdict(int)
        total = defaultdict(int)
        for section in ('dev', 'test'):
            length = len(self.data[section])
            for i in range(0, length, batch_size):

                batch = self.data[section][i:i + batch_size]
                ids, ids_lengths, labels = self.get_batch_full(batch, batch_size)
                feed_dict = dict({model.ids: ids, model.ids_lengths: ids_lengths})
                pred = session.run([model.pred], feed_dict)[0]

                for i, (pred, gd) in enumerate(zip(pred, labels)):
                    if len(batch) <= i:
                        break
                    total[section] += 1
                    if pred[0] == gd[0] and pred[2] == gd[2]:
                        c1 = 1
                    else:
                        c1 = 0
                    if pred[1] == gd[1] and pred[3] == gd[3]:
                        c2= 1
                    else:
                        c2 = 0
                    num_correct[section] += c1
                    batchi = batch[i]
                    rows = batchi.get("tags", [])
                    for tag in rows:
                        total[tag] += 1
                        num_correct[tag] += c1

        for section, correct in num_correct.items():
            print(section, format(correct / total[section], '.4f'), end='   ')  #
        return num_correct['english'] / total['english']


def train(args):
    if args.logdir is None:
        args.logdir = tempfile.mkdtemp(prefix='ifttt_')
        print(args.logdir)
    if args.clear and not (args.number_logdir or args.test_logdir):
        try:
            shutil.rmtree(args.logdir)
        except OSError:
            pass

    # for training...
    config = json.load(open(args.config, 'r'))
    print("configure: {}".format(config))
    tf_utils.mkdir_p(args.logdir)
    print("writing log to: {}".format(args.logdir))
    if args.number_logdir:
        sub_logdirs = os.listdir(args.logdir)
        sub_logdirs.sort()
        logdir_id = int(sub_logdirs[-1]) + 1 if sub_logdirs else 0
        args.logdir = os.path.join(args.logdir, '{:06d}'.format(logdir_id))
        print("Log dir is: {}".format(args.logdir))
        os.mkdir(args.logdir)

    with open(os.path.join(args.logdir, 'config.json'), 'w') as fw:
        print('dumping model config.json to: {}'.format(args.logdir))
        json.dump(config, fw, indent=4)

    try:
        # the training epoch
        mainloop = MainLoop(args, config)
        mainloop.run(mainloop.initializer, args.logdir, args.load_model)
    except KeyboardInterrupt:
        print("keyboard exception, training terminated!")
        pass
    print('Log dir: {}'.format(args.logdir))


def evaluate_test():
    print("Testing model with best parameters settings...")
    config = json.load(open(os.path.join(args.logdir, 'config.json')))
    stats = json.load(open(os.path.join(args.logdir, 'stats.json')))

    print("building MainLoop ...")
    ifttt_train = MainLoop(args, config)

    with tf.variable_scope('model', reuse=None, initializer=None):
        print('create model....')
        m = ifttt_train.create_model(is_train=False)

    for best_iter, name in zip(stats['best_iters'], stats['keys']):
        print(name)
        saver = tf.train.Saver(max_to_keep=0)
        print("running sessions")
        with tf.Session() as sess:
            print("load the best model after training...")
            saver.restore(sess, os.path.join(args.logdir, 'model.ckpt-{}'.format(int(best_iter))))
            # section type -> label type -> rows
            probs_by_section = defaultdict(lambda: [[] for i in range(4)])
            labels_by_section = defaultdict(lambda: [[] for i in range(4)])

            batch_size = ifttt_train.config['batch_size']
            for i in range(0, len(ifttt_train.data['test']), batch_size):
                batch = ifttt_train.data['test'][i:i + batch_size]

                all_probs = ifttt_train.eval_batch(sess, m, batch, batch_size, get_probs=True)
                assert len(all_probs) == 4

                for row_index, row in enumerate(batch):
                    for tag in row.get('tags', []):
                        tagged_section = 'test-{}'.format(tag)
                        for prob_matrix, container in zip(all_probs, probs_by_section[tagged_section]):
                            container.append(prob_matrix[row_index])

                        for label, container in zip(row['labels'], labels_by_section[tagged_section]):
                            container.append(label)
                sys.stdout.write(".")
                sys.stdout.flush()
            print(" ")

        with open(os.path.join(args.logdir, 'probs-{}.pkl'.format(name)), 'wb') as fw:
            data_dict = {'probs': dict(probs_by_section),
                         'labels': dict(labels_by_section)}
            pickle.dump(data_dict, fw, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--load-model')
    parser.add_argument('--bipartite-graph-file', default='./dataset/IFTTT/services-bipartite.txt')
    parser.add_argument('--services-file', default='./dataset/IFTTT/serv_name.txt')
    parser.add_argument('--functions-file', default='./dataset/IFTTT/functions.txt')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--config', required=True)
    parser.add_argument('--number-logdir', action='store_true')
    parser.add_argument('--test-logdir', action='store_true')
    parser.add_argument('--logdir')
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print("args", args)
    if args.mode == 'train':
        train(args)
    else:
        evaluate_test(args)
