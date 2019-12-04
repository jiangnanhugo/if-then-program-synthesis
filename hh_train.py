import argparse
import bisect
import pickle
from collections import defaultdict
import json
import random
import codecs
import os
import sys
import datetime
from tqdm import tqdm
import tensorflow as tf
import tf_utils
from tf_utils import IFTTTHHModel
import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


class MainLoop(object):
    def __init__(self, args, config,  maps, label_reindex):
        self.config = config
        self.data = pickle.load(open(args.dataset, 'rb'), encoding="latin1")
        self.label2index = self.data["label2index"]
        self.index2label = self.data["index2label"]
        self.maps = maps
        self.label_reindex = label_reindex

        self.max_desc_length = 0
        for item in (self.data['train'] + self.data['dev']):
            self.max_desc_length = max(self.max_desc_length, len(item['ids']))

        self.seq_len = self.config['seq_len']

        for section in ('train', 'test', 'dev'):
            for i in range(len(self.data[section])):
                if len(self.data[section][i]['ids']) > self.seq_len:
                    d = self.data[section][i]['ids']
                    d = d[:(self.seq_len + 1) // 2] + d[len(d) - self.seq_len//2:]
                    self.data[section][i]['ids'] = d

        buckets = tf_utils.create_buckets([len(item['ids']) for item in self.data['train']], self.config['batch_size'] * 5000)
        print('The bucket size:', buckets)

        self.bucketed_train = [[]] * len(buckets)
        for item in self.data['train']:
            size = len(item['ids'])
            self.bucketed_train[bisect.bisect_left(buckets, size)].append(item)

        bucketed_train_lens = np.array([len(bucket) for bucket in self.bucketed_train], dtype=float)
        self.bucket_dist = bucketed_train_lens / np.sum(bucketed_train_lens)

        scale = self.config['init']['scale']
        self.initializer = getattr(tf, self.config['init']['name'])(-scale, scale)

        self.vocab_size = int(self.config['max_word_id'])

    def read(self, cur_point):
        if cur_point == 0:
            random.shuffle(self.bucketed_train[0])

        bucket = self.bucketed_train[0]
        batch = bucket[cur_point:cur_point + self.config['batch_size']]
        batch = batch + [random.choice(bucket) for _ in range(self.config['batch_size'] - len(batch))]
        return self.get_batch(batch)

    def get_batch(self, batch):
        ids = tf_utils.make_array([item['ids'] for item in batch], length=self.seq_len)     # [batch_size(32), max_seq_len(25)]
        ids_lengths = np.array([len(item['ids']) for item in batch])                            # [batch_size(32)]
        top_labels = np.zeros((self.config['batch_size'], 2), dtype=np.int32)                   # [batch_size(32), label types(2)]
        bot_labels = np.zeros((self.config['batch_size'], 2), dtype=np.int32)                   # [batch_size(32), label types(4)]
        for i, item in enumerate(batch):
            lab = [x.lower() for x in item['label_names']]
            lab = [self.label2index[j][lab[j]] for j in range(4)]  # for ext exp

            top_labels[i, 0] = lab[self.label_reindex[0]]
            top_labels[i, 1] = lab[self.label_reindex[1]]
            per_class = self.maps[0]['mapping'][top_labels[i, 0]]
            bot_labels[i, 0] = per_class.index(lab[self.label_reindex[2]])
            per_class = self.maps[1]['mapping'][top_labels[i, 1]]
            bot_labels[i, 1] = per_class.index(lab[self.label_reindex[3]])

        return ids, ids_lengths, top_labels, bot_labels

    def get_batch_with_labelname(self, batch, batch_size):
        ids = tf_utils.make_array([item['ids'] for item in batch], length=self.seq_len, batch_size=batch_size)
        ids_lengths = np.zeros(batch_size)
        for i, item in enumerate(batch):
            ids_lengths[i] = len(item['ids'])
        labels = []
        for item in batch:
            lab = [x.lower() for x in item['label_names']]
            labels.append(lab)

        return ids, ids_lengths, labels

    def create_model(self, is_train, stop_gradient=False):
        return IFTTTHHModel(self.config, self.vocab_size, self.seq_len, self.maps, is_train, stop_gradient)

    def run(self, initializer, load_model=None):
        epoches = self.config['epoches']
        batch_size = self.config['batch_size']
        with tf.variable_scope('model', reuse=None, initializer=initializer):
            model = self.create_model(is_train=True, stop_gradient=(load_model is not None))
        with tf.variable_scope('model', reuse=True, initializer=initializer):
            model_valid = self.create_model(is_train=False)
        saver = tf.train.Saver(max_to_keep=0)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        session = tf.Session(config=config)
        # intialize the params
        if load_model:
            saver.restore(session, os.path.join("model", 'hhmodel.ckpt-{}'.format(load_model)))
            self.generate_test_summaries(session, model_valid, print_detail=True)
            return
        else:
            session.run(tf.global_variables_initializer())
        print()
        epoch = 0
        cur_point = 0
        best_acc = 0
        while epoch <= epoches:
            epoch += 1
            iterations = int(len(self.bucketed_train[0]) / batch_size)
            loss = 0.
            acc = 0.
            for i in tqdm(range(iterations), leave=False):
                batch = self.read(cur_point)
                ids, ids_lengths, top, bottom = batch
                cur_point += len(batch[0])
                # Run 1 step with mini-batch
                input_list = [model.loss, model.train_op, model.optimizer._lr, model.global_step]

                feed_dict = dict({model.ids: ids,
                                  model.ids_lengths: ids_lengths,
                                  model.y_top: top,
                                  model.y_bottom: bottom})
                cost, _, cur_lr, global_step = session.run(input_list, feed_dict)
                loss += np.sum(cost)
            # reset data pointer
            cur_point = 0
            print("epoch:", epoch,
                  " trainLoss:", format(loss*batch_size/len(self.bucketed_train[0]), '.4f'), end='  ')
                  # " train_acc:", acc,
            cur_acc = self.generate_test_summaries(session, model_valid)

            print('  lr:', format(cur_lr, '.6f'), end='  ')
            if cur_acc > best_acc:
                best_acc = cur_acc
                print("save acc:", format(cur_acc, '.4f'), 'T:', datetime.datetime.now().isoformat(timespec='minutes'), end='  ')
                # saver.save(session, "./model/hhmodel.ckpt", global_step=global_step)
            print()


    def generate_test_summaries(self, session, model,print_detail=False):
        batch_size = self.config['batch_size']
        num_correct = defaultdict(lambda: np.zeros(5, dtype=np.float))
        total = defaultdict(int)
        bmap = model.bmap
        for section in ('dev', 'test'):
            length = len(self.data[section])
            for i in range(0, length, batch_size):
                batch = self.data[section][i:i + batch_size]
                ids, ids_lengths, gold_names = self.get_batch_with_labelname(batch, batch_size)

                feed_dict = dict({model.ids: ids, model.ids_lengths: ids_lengths})
                pred_top, pred_bot = session.run([model.pred_top, model.pred_bottom], feed_dict)
                pred_reidx_bot = []
                for x, y in zip(pred_top, pred_bot):
                    pred_reidx_bot.append([bmap[0][x[0]][y[0]], bmap[1][x[1]][y[1]]])

                # print('pred_bot:', pred_bot.flatten(), 'pred_reindex_bottom:', pred_reidx_bot)
                for j, (pred_top_idx, pred_bot_idx, gold_name) in \
                        enumerate(zip(pred_top, pred_reidx_bot, gold_names)):
                    if j >= len(batch):
                        break
                    total[section] += 1
                    pred_name = ["", "", "", ""]

                    index2label = self.index2label[self.label_reindex[0]]
                    name = index2label[pred_top_idx[0]]
                    pred_name[self.label_reindex[0]] = name

                    index2label = self.index2label[self.label_reindex[1]]
                    name=index2label[pred_top_idx[1]]
                    pred_name[self.label_reindex[1]] = name

                    index2label = self.index2label[self.label_reindex[2]]
                    # print(len(index2label), end=" ")
                    name=index2label[pred_bot_idx[0]]
                    pred_name[self.label_reindex[2]] = name

                    index2label = self.index2label[self.label_reindex[3]]
                    # print(len(index2label))
                    name=index2label[pred_bot_idx[1]]
                    pred_name[self.label_reindex[3]] = name

                    # print(pred_name, gold_name)
                    matched = [(x == y) for x, y in zip(pred_name, gold_name)]
                    matched.append(pred_name == gold_name)
                    num_correct[section] += matched
                    rows = batch[j].get("tags", [])
                    for tag in rows:
                        total[tag] += 1
                        num_correct[tag] += matched

        for section, correct in num_correct.items():
            if section.startswith("intelligible"):
                five_parts = correct / total[section]
                for k in five_parts:
                    print(format(k, '.4f'), end='  ')
        sys.stdout.flush()

        return num_correct['intelligible'][-1] / total['intelligible']


def train(args, choices):
    config = json.load(open(args.config, 'r'))
    print("Configure:\n ", config)
    dict_file = config['dict_file']
    maps = []

    choices = choices[args.model_choice]
    label_reindex = choices[0]
    for choice in choices[-1]:
        map = defaultdict()
        with codecs.open(dict_file[choice], 'r', 'utf-8')as fr:
            lines = fr.read().split('\n')
            shape = lines[0].split(' ')
            map['shape'] = (int(shape[0]), int(shape[1]), int(shape[2]))
            tuple_list = []
            for i in range(1, len(lines)):
                line = lines[i].strip().split(" ")
                if len(line) == 2:
                    tuple_list.append([int(line[0]), int(line[1])])
            bipartite = []
            i = 0
            count = 0
            while i < len(tuple_list):
                j = i + 1
                while j < len(tuple_list) and tuple_list[j][0] == tuple_list[i][0]:
                    j += 1
                if j == i+1:
                    count += 1
                bipartite.append([x[1] for x in tuple_list[i:j]])
                i = j
            # print("count: {}, all: {}".format(count, len(tuple_list)))
            map['mapping'] = bipartite
        maps.append(map)

    try:
        # print([label_reindex[x] for x in range(4)])
        # print([label_reindex.index(x) for x in range(4)])
        mainloop = MainLoop(args, config, maps, label_reindex)
        mainloop.run(mainloop.initializer, args.load_model)
    except KeyboardInterrupt:
        print("keyboard exception, training terminated!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="./dataset/IFTTT/msr_data.pkl")
    parser.add_argument('--load-model')
    parser.add_argument('--model-choice', default='0')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    model_choice = {
        '0': [(0, 2, 1, 3), ('zero_one_bipartite', 'two_three_bipartite')],  # (label[0]->label[2]), (label[1]->label[3])
        '1': [(0, 3, 1, 2), ('zero_one_bipartite', 'three_two_bipartite')],
        '2': [(0, 1, 2, 3), ('zero_two_bipartite', 'one_three_bipartite')],   # (label[0]->label[2]), (label[3]->label[1])
        '3': [(0, 3, 2, 1), ('zero_two_bipartite', 'three_one_bipartite')],  # (label[0]->label[1]), (label[2]->label[3])
        '4': [(0, 1, 3, 2), ('zero_three_bipartite', 'one_two_bipartite')],
        '5': [(0, 2, 3, 1), ('zero_three_bipartite', 'two_one_bipartite')],
        '6': [(1, 2, 0, 3), ('one_zero_bipartite', 'two_three_bipartite')],
        '7': [(1, 3, 0, 2), ('one_zero_bipartite', 'three_two_bipartite')],
        '8': [(2, 1, 0, 3), ('two_zero_bipartite', 'one_three_bipartite')],
        '9': [(2, 3, 0, 1), ('two_zero_bipartite', 'three_one_bipartite')],
        '10': [(3, 1, 0, 2), ('three_zero_bipartite', 'one_two_bipartite')],
        '11': [(3, 2, 0, 1), ('three_zero_bipartite', 'two_one_bipartite')],
    }
    train(args, model_choice)
