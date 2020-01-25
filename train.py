import argparse
import bisect
import pickle
from collections import defaultdict
import json
import random
import os
import datetime
from tqdm import tqdm
import tensorflow as tf
import timeit
import tf_utils
from tf_utils import Model, get_vps
import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=np.inf)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.set_random_seed(10086)
np.random.seed(10010)
random.seed(47906)


class MainLoop(object):
    def __init__(self, args, config, vps_max_width):
        self.config = config
        self.output_dir = args.output
        self.data = pickle.load(open(args.dataset, 'rb'), encoding="latin1")
        self.max_desc_length = 0

        all_labels = self.get_label_instance(self.data)
        self.label_set=[defaultdict(set),defaultdict(set)]
        for i in range(len(all_labels)):
            self.label_set[0][all_labels[i][0]].add(all_labels[i][1])
            self.label_set[1][all_labels[i][2]].add(all_labels[i][3])

        self.vps = get_vps(all_labels, vps_max_width, self.output_dir, "dataset", self.data['category'])

        self.label2index = self.data['label2index']
        self.index2label = self.data['index2label']
        self.label_size = [len(x) for x in self.data['index2label']]

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

    @staticmethod
    def get_label_instance(datadict):
        names = [it['label_names'] for it in datadict['train']]
        names += [it['label_names'] for it in datadict['dev']]
        label_names = []
        for na in names:
            na = [x.lower() for x in na]
            if na not in label_names:
                label_names.append(na)
        print("# path", len(label_names))
        return label_names

    def get_mask(self, label_names):
        neighbors = self.vps.find_neighbor_along_path(label_names)
        batch_mask = []
        for i in range(4):
            m = np.zeros(shape=self.label_size[i], dtype=np.float32)
            for neib in neighbors[i]:
                wid = self.label2index[i][neib]
                m[wid] = 1.0
            batch_mask.append(m)
        return batch_mask

    def update_mask_with_depth(self, label_names, w):
        batch_mask = [np.ones((self.config['batch_size'], self.label_size[ii]), dtype=np.int32) for ii in range(4)]
        for j in range(len(label_names)):
            new_labels = label_names[j][:w]
            neighbors = self.vps.find_last_neighbor_along_path(new_labels)
            for i in range(4):
                if i <= w:
                    m = np.zeros(shape=self.label_size[i], dtype=np.float32)
                    for neib in neighbors[i]:
                        wid = self.label2index[i][neib]
                        m[wid] = 1.0
                else:
                    m = np.ones(shape=self.label_size[i], dtype=np.float32)
                batch_mask[i][j, :] = m
        return batch_mask

    def read(self, cur_point):
        if cur_point == 0:
            random.shuffle(self.bucketed_train[0])

        bucket = self.bucketed_train[0]
        batch = bucket[cur_point:cur_point + self.config['batch_size']]
        batch = batch + [random.choice(bucket)
                         for _ in range(self.config['batch_size'] - len(batch))]
        return self.get_batch(batch)

    def get_batch(self, batch):
        ids = tf_utils.make_array([item['ids'] for item in batch], length=self.memory_size)
        ids_lengths = np.array([len(item['ids']) for item in batch])                         # [batch_size(32)]
        labels = np.zeros((self.config['batch_size'], 4), dtype=np.int32)
        batch_mask = [np.zeros((self.config['batch_size'], self.label_size[i]), dtype=np.int32) for i in range(4)]
        for i, item in enumerate(batch):
            lab = [x.lower() for x in item['label_names']]
            lab = [self.label2index[j][lab[j]] for j in range(4)]  # for ext exp
            labels[i, :] = lab
            mask = self.get_mask(item['label_names'])
            for j in range(4):
                batch_mask[j][i, :] = mask[j]

        return ids, ids_lengths, labels, batch_mask

    def get_batch_full(self, batch, batch_size):
        ids = tf_utils.make_array([item['ids'] for item in batch], length=self.memory_size, batch_size=batch_size)
        ids_lengths = np.zeros(batch_size)
        for i, item in enumerate(batch):
            ids_lengths[i] = len(item['ids'])
        labels = []
        for i, item in enumerate(batch):
            lab = [x.lower() for x in item['label_names']]
            labels.append(lab)

        return ids, ids_lengths, labels

    def create_model(self, is_train):
        return Model(self.config, self.vocab_size, self.memory_size, self.label_size, is_train)

    def run(self, initializer, load_model=None):
        epoches = self.config['epoches']
        batch_size = self.config['batch_size']
        with tf.variable_scope('model', reuse=None, initializer=initializer):
            model = self.create_model(is_train=True)
        with tf.variable_scope('model', reuse=True, initializer=initializer):
            model_valid = self.create_model(is_train=False)
        saver = tf.train.Saver(max_to_keep=0)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        session = tf.Session(config=config)

        if load_model:
            saver.restore(session, os.path.join(args.logdir, 'model.ckpt-{}'.format(load_model)))
        else:
            session.run(tf.global_variables_initializer())

        epoch, cur_point, best_acc = 0, 0, 0
        print("the graph width list : {}".format(self.vps.widthList))
        while epoch <= epoches:
            epoch += 1
            iterations = int(len(self.bucketed_train[0]) / batch_size)
            # timed=0
            #for _ in tqdm(range(iterations), leave=False):
            for _ in range(iterations):
                batch = self.read(cur_point)
                cur_point += len(batch[0])
                input_list = [model.train_op, model.global_step]
                ids, ids_lengths, label, batch_mask = batch
                # for i in range(32):
                #     for j in range(4):
                #          print("i={}, j={}, mask={}".format(i, j, np.sum(batch_mask[j][i])))

                feed_dict = dict({model.ids: ids,
                                  model.ids_lengths: ids_lengths,
                                  model.label: label,
                                  model.batch_mask[0]: batch_mask[0],
                                  model.batch_mask[1]: batch_mask[1],
                                  model.batch_mask[2]: batch_mask[2],
                                  model.batch_mask[3]: batch_mask[3]})
                # start = timeit.default_timer()
                train_op, global_step = session.run(input_list, feed_dict)
                # timed+=end-start
            # print("timed {}",timed)
            # reset data pointer
            cur_point = 0
            print("it", epoch, end=" ")

            cur_acc, int_acc, tp_instance = self.generate_test_summaries(session, model_valid)
            if cur_acc > best_acc:
                best_acc = cur_acc
                print("save:", format(int_acc, '.2f'),
                      'T:', datetime.datetime.now().isoformat(timespec='minutes'), end=' ')
            print()

    def generate_test_summaries(self, session, model):
        batch_size = self.config['batch_size']
        num_correct = defaultdict(int)
        total = defaultdict(int)
        tp_intance=defaultdict(list)
        if 'new_test' in self.data:
            candidate=('dev', 'new_test')
        else:
            candidate=('dev', 'test')

        for section in candidate:
            tp_intance[section] = []
            length = len(self.data[section])
            for i in range(0, length, batch_size):
                batch = self.data[section][i:i + batch_size]
                ids, ids_lengths, labels = self.get_batch_full(batch, batch_size)
                batch_mask = [np.ones((batch_size, self.label_size[k]), dtype=np.int32) for k in range(4)]
                for iter in range(1, 5):
                    feed_dict = dict({model.ids: ids,
                                      model.ids_lengths: ids_lengths,
                                      model.batch_mask[0]: batch_mask[0],
                                      model.batch_mask[1]: batch_mask[1],
                                      model.batch_mask[2]: batch_mask[2],
                                      model.batch_mask[3]: batch_mask[3]})

                    pred = session.run(model.pred, feed_dict)
                    predict_names = []
                    for t in range(len(pred)):
                        predict = []
                        for j in range(iter):
                            lab_name = self.index2label[j][pred[t][j]]
                            predict.append(lab_name)
                        predict_names.append(predict)
                    # print(predict_names)
                    batch_mask = self.update_mask_with_depth(predict_names, iter)

                for j, (pd, gd) in enumerate(zip(predict_names, labels)):
                    if len(batch) <= j:
                        break
                    if pd[0] == gd[0] and pd[1] == gd[1] and pd[2] == gd[2] and pd[3] == gd[3]:
                        c1 = 1
                    else:
                        c1 = 0

                    if pd[3] in self.label_set[1][pd[2]] and pd[1] in self.label_set[0][pd[0]]:
                        c3=1
                    else:
                        c3=0
                        # print(pd_name[2], pd_name[3], self.label_set[pd_name[2]])
                    tp_intance[section].append((batch[j], predict_names[j]))
                    c2 = ((pd[0] != gd[0]) + (pd[1] != gd[1]) + (pd[2] != gd[2]) + (pd[3] != gd[3]))*0.25
                    total[section] += 1
                    total[section+"_h"] += 1
                    total[section + "_v"] += 1
                    num_correct[section] += c1
                    num_correct[section+"_h"] += c2
                    num_correct[section + "_v"] += c3

        for section, correct in num_correct.items():
            print(section, format(correct*100 / total[section], '.2f'), end='   ')

        return num_correct['dev'] / total['dev'], num_correct[candidate[-1]]*100 / total[candidate[-1]], tp_intance[candidate[-1]]




def train(args):

    # for training...
    config = json.load(open(args.config, 'r'))
    print("configure: {}".format(config))


    try:
        mainloop = MainLoop(args, config, args.width)
        mainloop.run(mainloop.initializer, args.load_model)
    except KeyboardInterrupt:
        print("keyboard exception, training terminated!")
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--width', default='0')
    parser.add_argument('--config', required=True)
    parser.add_argument('--load-model')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print("args", args)
    if args.mode == 'train':
        train(args)
