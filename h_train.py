import argparse
import bisect
import pickle
from collections import defaultdict
import json
import random
import codecs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import sys
from tqdm import tqdm
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import tf_utils
from tf_utils import TFMainLoop, IFTTTHModel
import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)


class MainLoop(TFMainLoop):
    def __init__(self, args, config, serv_dict, func_dict, b_graph):
        self.config = config
        self.root_logdir = args.logdir
        self.output_file = args.output
        self.serv_dict = serv_dict
        self.inv_serv_dict = {v: k for k, v in serv_dict.items()}
        self.func_dict = func_dict
        self.inv_func_dict = {v: k for k, v in func_dict.items()}
        self.b_graph = b_graph
        print("top layer size: {}, bottom layer size: {}".format(self.config['top_size'], self.config['bottom_size']))
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
        # print('The bucket size:', buckets)

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
        # self.get_batch_with_labelname(batch,self.config['batch_size'])
        return self.get_batch(batch)

    def get_batch(self, batch):
        # [batch_size(32), max_seq_len(25)]
        ids = tf_utils.make_array([item['ids'] for item in batch], length=self.memory_size)
        # [batch_size(32)]
        ids_lengths = np.array([len(item['ids']) for item in batch])
        # [batch_size(32), label types(4)]
        top_labels = np.zeros(self.config['batch_size'], dtype=np.int32)
        bottom_labels = np.zeros(self.config['batch_size'], dtype=np.int32)
        y_ids = np.zeros(self.config['batch_size'],dtype=np.int32)
        for i, item in enumerate(batch):
            try:
                s1, f1, s2, f2 = [x.decode('utf-8') for x in item['label_names']]
                x_id = self.serv_dict[(s1, s2)]
                y_id = self.func_dict[(f1.split(".")[-1], f2.split(".")[-1])]
                if (x_id is None) and (y_id is None):
                    continue
                per_class = self.b_graph[x_id]
                in_class_y = per_class.index(y_id)
                top_labels[i] = x_id
                y_ids[i] = y_id
                bottom_labels[i] = in_class_y
            except:
                print("error")

        return ids, ids_lengths, top_labels, bottom_labels, y_ids

    def get_batch_with_labelname(self, batch, batch_size):
        ids = tf_utils.make_array([item['ids'] for item in batch], length=self.memory_size, batch_size=batch_size)
        ids_lengths = np.zeros(batch_size)
        for i, item in enumerate(batch):
            ids_lengths[i] = len(item['ids'])
        top_labels = []
        bottom_labels = []
        for i, item in enumerate(batch):
            s1, f1, s2, f2 = [x.decode('utf-8') for x in item['label_names']]
            f1, f2 = f1.split(".")[-1], f2.split(".")[-1]
            top_labels.append((s1, s2))
            bottom_labels.append((f1, f2))

        return ids, ids_lengths, top_labels, bottom_labels

    def create_model(self, is_train, stop_gradient=False):
        return IFTTTHModel(self.config, self.vocab_size, self.memory_size, self.b_graph, is_train, stop_gradient )

    def run(self, initializer, logdir, load_model=None):
        epoches = self.config['epoches']
        batch_size = self.config['batch_size']
        with tf.variable_scope('model', reuse=None, initializer=initializer):
            model = self.create_model(is_train=True, stop_gradient=(load_model is not None))

        with tf.variable_scope('model', reuse=True, initializer=initializer):
            model_valid = self.create_model(is_train=False)
        saver = tf.train.Saver(max_to_keep=0)
        session = tf.Session()
        # intialize the params
        if load_model:
            saver.restore(session, os.path.join(args.logdir, 'model.ckpt-{}'.format(load_model)))
            self.generate_test_summaries(session, model_valid)
            return
            initial_global_step = model.global_step.assign(50)
            session.run(initial_global_step)
        else:
            session.run(tf.global_variables_initializer())
        # Disable automatic saving and deleting
        supervisor = tf.train.Supervisor(logdir=logdir, summary_op=None, save_model_secs=0, saver=saver)
        self.initialize_extra_vars(session, model)
        # self.generate_test_summaries(session, model_valid)
        print()
        epoch = 0
        cur_point = 0
        best_intel_acc = 0
        while not supervisor.should_stop() and epoch <= epoches:
            epoch += 1
            iterations = int(len(self.bucketed_train[0]) / batch_size)
            loss = 0.
            for _ in tqdm(range(iterations), leave=False):
                batch = self.read(cur_point)
                cur_point += len(batch[0])
                # Run 1 step with mini-batch
                input_list = [model.xentropy, model.bipart, model.train_op, model.optimizer._lr, model.global_step]
                ids, ids_lengths, top, bottom, y_ids = batch
                feed_dict = dict({model.ids: ids,
                                  model.ids_lengths: ids_lengths,
                                  model.y_top: top,
                                  model.y_bottom: bottom})
                cost, cur_bipart, _, cur_lr, global_step = session.run(input_list, feed_dict)
                loss += np.sum(cost)

                # break
            # reset data pointer
            cur_point = 0
            print("epoch:", epoch,
                  # "train:", format(c1*1./len(self.bucketed_train[0]), '.4f'),
                  " train_loss:", format(loss/len(self.bucketed_train[0]), '.4f'), end='\t')
            intel_acc = self.generate_test_summaries(session, model_valid)
            # break
            print('\tlr:', format(cur_lr, '.6f'), end='    ')
            if intel_acc > best_intel_acc:
                best_intel_acc = intel_acc
                print("save acc:", intel_acc, 'T:', datetime.datetime.now(), end=' ')
                supervisor.saver.save(session, supervisor.save_path, global_step=global_step)
            print()

    def generate_test_summaries(self, session, model):
        batch_size = self.config['batch_size']
        num_correct = defaultdict(int)
        total = defaultdict(int)
        b_mapping = model.b_mapping
        for section in ('dev', 'test'):
            length = len(self.data[section])
            for i in range(0, length, batch_size):
                batch = self.data[section][i:i + batch_size]
                ids, ids_lengths, top_names, bot_names = self.get_batch_with_labelname(batch, batch_size)
                feed_dict = dict({model.ids: ids, model.ids_lengths: ids_lengths})
                pred_top, pred_bot = session.run([model.pred_y_top, model.pred_y_bottom], feed_dict)
                pred_bot = [b_mapping[(x, y)] for x, y in zip(pred_top, pred_bot)]
                # print('pred_t:', pred_top, 'pred_bottom:', pred_bot)
                for j, (pred_top_idx, pred_bot_idx, gold_top_name, gold_bot_name) in \
                        enumerate(zip(pred_top, pred_bot, top_names, bot_names)):
                    if j >= len(batch):
                        break
                    total[section] += 1
                    pred_top_name = self.inv_serv_dict[pred_top_idx]
                    # bot_index = b_mapping[pred_top_idx]
                    pred_bot_name = self.inv_func_dict[pred_bot_idx]
                    c1 = (pred_top_name == gold_top_name)  # predict the four label together.
                    c2 = (pred_bot_name == gold_bot_name)

                    rows = batch[j].get("tags", [])
                    for tag in rows:
                        total[tag] += 1
                        num_correct['t_'+tag] += c1
                        # total['b_' + tag] += 1
                        num_correct['b_' + tag] += c2
                        if c1 == 0 and tag == 'gold':  # and c2 == 0:
                            pass
                            # print(" ".join(batch[j]['words']), pred_top_name,
                            #       gold_top_name)  # , pred_bot_name, gold_bot_name)
                        # num_correct[section] += c1

        # for section, correct in num_correct.items():
        #     print(section, format(correct / total[section], '.4f'), end='   ')
        t_intel_acc = format(num_correct['t_intelligible'] / total['intelligible'], '.4f')
        b_intel_acc = format(num_correct['b_intelligible'] / total['intelligible'], '.4f')
        # intel_eng_acc = format(num_correct['intel+eng'] / total['intel+eng'], '.4f')
        #print('t_intell', t_intel_acc, )
        print('b_intell:', b_intel_acc, end='   ')
        t_intel_acc = format(num_correct['t_english'] / total['english'], '.4f')
        b_intel_acc = format(num_correct['b_english'] / total['english'], '.4f')
        # intel_eng_acc = format(num_correct['intel+eng'] / total['intel+eng'], '.4f')
        #print('t_english', t_intel_acc, #
        print('b_english:', b_intel_acc, end='   ')
        t_intel_acc = format(num_correct['t_gold'] / total['gold'], '.4f')
        b_intel_acc = format(num_correct['b_gold'] / total['gold'], '.4f')
        # intel_eng_acc = format(num_correct['intel+eng'] / total['intel+eng'], '.4f')
        #print('t_gold', t_intel_acc,
        print('b_gold:', b_intel_acc, end='   ')

        return float(t_intel_acc)


def train(args):
    with codecs.open(args.services_file, 'r', 'utf-8')as fr:
        lines = fr.read().split('\n')
        services = defaultdict(tuple)
        for l in lines:
            f = l.strip().split(' ')
            if len(f) == 3:
                services[(f[0], f[1])] = int(f[2])

    with codecs.open(args.functions_file, 'r', 'utf-8')as fr:
        lines = fr.read().split('\n')
        functions = defaultdict(tuple)
        for l in lines:
            s = l.strip().split(' ')
            if len(s) == 3:
                functions[(s[0], s[1])] = int(s[2])

    with codecs.open(args.bipartite_graph_file, 'r', 'utf-8')as fr:
        lines = fr.read().split('\n')
        bipt_size = lines[0].split(' ')
        bipartite_list = []
        for i in range(1, len(lines)):
            xy = lines[i].strip().split(" ")
            if len(xy) == 2:
                bipartite_list.append([int(xy[0]), int(xy[1])])
        bipartite = []
        i = 0
        count = 0
        while i < len(bipartite_list):
            j = i + 1
            while j < len(bipartite_list) and bipartite_list[j][0] == bipartite_list[i][0]:
                j += 1
            if j == i+1:
                count += 1
            bipartite.append([x[1] for x in bipartite_list[i:j]])
            i = j
        print("count: {}, all: {}".format(count, len(bipartite_list)))

    # for training...
    config = json.load(open(args.config, 'r'))
    config['top_size'] = int(bipt_size[0])
    config['bottom_size'] = int(bipt_size[1])
    print("configure:\n {}".format(config))

    try:
        # the training epoch
        mainloop = MainLoop(args, config, services, functions, bipartite)
        mainloop.run(mainloop.initializer, args.logdir, args.load_model)
    except KeyboardInterrupt:
        print("keyboard exception, training terminated!")
        pass
    print('Log dir: {}'.format(args.logdir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--load-model')
    parser.add_argument('--bipartite-graph-file', default='./dataset/IFTTT/service-function-bipartite.txt')
    parser.add_argument('--services-file', default='./dataset/IFTTT/services.txt')
    parser.add_argument('--functions-file', default='./dataset/IFTTT/functions.txt')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--config', required=True)
    parser.add_argument('--number-logdir', action='store_true')
    parser.add_argument('--test-logdir', action='store_true')
    parser.add_argument('--logdir')
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    # tf_utils.test_h_softmax()
    if args.mode == 'train':
        train(args)
    else:
        evaluate_test(args)
