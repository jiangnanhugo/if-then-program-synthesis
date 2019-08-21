import argparse, bisect
import pickle
from collections import defaultdict
import json, random
import os, shutil, sys
import datetime
import tempfile
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import tf_utils
from tf_utils import IFTTTModel
import codecs
np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


class MainLoop(object):
    def __init__(self, args, config, label2index, index2label):
        self.config = config
        self.optim_config = config['optim']
        self.model_config = config['model']
        self.eval_config = config['eval']
        self.arch_config = config.get('architecture', None)
        self.root_logdir = args.logdir
        self.output_file = args.output
        self.label2index = label2index
        self.index2label = index2label

        self.data = pickle.load(open(args.dataset, 'rb'), encoding="latin1")
        if 'label_types' in self.data:
            global label_names
            label_names = self.data['label_types']
        self.max_desc_length = 0

        for item in (self.data['train'] + self.data['dev'] + self.data['test']):
            self.max_desc_length = max(self.max_desc_length, len(item['ids']))

        self.memory_size = self.model_config.get('memory_size', self.max_desc_length)

        for section in ('train', 'test', 'dev'):
            for i in range(len(self.data[section])):
                if len(self.data[section][i]['ids']) > self.memory_size:
                    d = self.data[section][i]['ids']
                    d = d[:(self.memory_size + 1) // 2] + d[len(d) - self.memory_size//2:]
                    self.data[section][i]['ids'] = d

        buckets = tf_utils.create_buckets([len(item['ids']) for item in self.data['train']],
                                            self.optim_config['batch_size'] * 5000)
        print('The bucket size:', buckets)

        self.bucketed_train = [[]] * len(buckets)
        for item in self.data['train']:
            size = len(item['ids'])
            self.bucketed_train[bisect.bisect_left(buckets, size)].append(item)

        bucketed_train_lens = np.array([len(bucket) for bucket in self.bucketed_train], dtype=float)
        self.bucket_dist = bucketed_train_lens / np.sum(bucketed_train_lens)

        scale = self.optim_config['init']['scale']
        self.initializer = getattr(tf, self.optim_config['init']['name'])(-scale, scale)

        max_word_id = int(self.arch_config['max_word_id'] if self.arch_config else
                          config['max_word_id'])
        if max_word_id > 0:
            self.vocab_size = max_word_id
        else:
            self.vocab_size = len(self.data['word_ids'])

        # Eval measures
        self.label_types = list(self.arch_config['label_types'] if self.arch_config
                                else self.config['label_types'])
        self.label_type_names = list(np.array(label_names)[self.label_types])

        ## what is this!
        if self.label_types == [0, 1, 2, 3]:
            self.label_type_names.append('tc+ac')
            self.label_type_names.append('tc+tf+ac+af')

        self.acc_by_section = defaultdict(int)
        self.best_accuracy = np.zeros(len(self.label_type_names))
        self.best_iters = np.zeros(len(self.label_type_names))

    def read_batch(self, cur_point):
        if cur_point == 0:
            random.shuffle(self.bucketed_train[0])

        bucket = self.bucketed_train[0]
        batch = bucket[cur_point:cur_point + self.optim_config['batch_size']]
        batch = batch + [random.choice(bucket)
                         for _ in range(self.optim_config['batch_size'] - len(batch))]
        # [batch_size(32), max_seq_len(25)]
        ids = tf_utils.make_array([item['ids'] for item in batch], length=self.memory_size)
        # [batch_size(32)]
        ids_lengths = np.array([len(item['ids']) for item in batch])
        # [batch_size(32), label types(4)]
        labels = tf_utils.make_array([item['labels'] for item in batch])[:, self.label_types]
        # batch_size(32), max_seq_len(25), label types(4)
        labels = np.zeros((len(batch), 4), dtype=np.int32)  # [batch_size(32), label types(4)]
        for i, item in enumerate(batch):
            try:
                lab = [x for x in item['label_names']]
                # lab[1], lab[3] = lab[1].split(".")[-1], lab[3].split(".")[-1]
                lab = [self.label2index[i][lab[i]] for i in range(4)]
                labels[i, :] = lab
            except:
                print("error", i)

        return ids, ids_lengths, labels

    def train_feed_dict(self, m, data):
        ids, ids_lengths, labels = data

        feed_dict = tf_utils.filter_none_keys({m.ids: ids, m.ids_lengths: ids_lengths,  m.labels: labels})
        return feed_dict

    def create_model(self, is_train):
        if self.arch_config:
            label_types = list(self.arch_config['label_types'])
        else:
            label_types = list(self.label_types)
        return IFTTTModel(self.optim_config,
                          self.model_config,
                          self.arch_config,
                          np.array(self.data['num_labels'])[label_types],
                          self.label_types,
                          self.vocab_size,
                          self.memory_size,
                          is_train)

    def run(self, initializer, logdir, load_model=None, label_embd=None):
        with tf.variable_scope('model', reuse=None, initializer=initializer):
            model = self.create_model(is_train=True)
        with tf.variable_scope('model', reuse=True, initializer=initializer):
            model_valid = self.create_model(is_train=False)
        saver = tf.train.Saver(max_to_keep=0)
        session = tf.Session()
        # intialize the params
        if load_model:
            saver.restore(session, load_model)
        else:
            session.run(tf.global_variables_initializer())
        if len(label_embd) != 0:
            session.run(tf.assign(model.softmax_placehold[0], label_embd[0]))
            session.run(tf.assign(model.softmax_placehold[1], label_embd[1]))
            session.run(tf.assign(model.softmax_placehold[2], label_embd[2]))
            session.run(tf.assign(model.softmax_placehold[3], label_embd[3]))
            print('loading pretrained label embedding...')
        # Disable automatic saving and deleting
        supervisor = tf.train.Supervisor(logdir=logdir, summary_op=None, save_model_secs=0, saver=saver)

        epoch = 0
        cur_point = 0
        best_acc = 0.
        print("length of bucket {}.....".format(len(self.bucketed_train)))
        while not supervisor.should_stop() and epoch <= self.optim_config['epoches']:
            epoch += 1
            print('epoch:', epoch, end='   ')
            iterations = int(len(self.bucketed_train[0]) / self.optim_config['batch_size'])

            for _ in tqdm(range(iterations), leave=False):
                batch = self.read_batch(cur_point)
                cur_point += len(batch[0])
                # Run 1 step with minibatch
                input_list = [model.loss, model.train_op, model.global_step]
                cost, _, global_step = session.run(input_list, self.train_feed_dict(model, batch))
                # break
            # reset data pointer
            cur_point = 0
            cur_acc = self.generate_test_summaries(session, model_valid)
            # Save model
            if cur_acc > best_acc:
                best_acc = cur_acc
                now = datetime.datetime.now()
                print("save acc:", format(cur_acc, '.4f'), 'T:', now.strftime("%y-%m-%d %H:%m"), end=' ')
            print()

    def generate_test_summaries(self, session, mtest):
        num_correct = defaultdict(lambda: np.zeros(len(self.label_type_names), dtype=np.float))
        total = defaultdict(int)
        batch_size = self.optim_config['batch_size']
        for section in ('dev', 'test'):
            for i in range(0, len(self.data[section]), batch_size):
                batch = self.data[section][i:i + batch_size]
                nc, _, _ = self.eval_batch(session, mtest, batch, batch_size)

                for j, row in enumerate(batch):
                    total[section] += 1
                    num_correct[section] += nc[j]
                    for tag in row.get('tags', []):
                        total[tag] += 1
                        num_correct[tag] += nc[j]

        for section, correct in num_correct.items():
            self.acc_by_section[section] = correct / total[section]
            print(section, format(self.acc_by_section[section][5], '.4f'), end='  ')
        # print()
        return self.acc_by_section['intelligible'][5]

    def eval_batch(self, session, mtest, batch, batch_size, get_probs=False):
        batch_size_original = len(batch)
        ids = tf_utils.make_array([item['ids'] for item in batch] + [batch[0]['ids'] \
                                  for _ in range(batch_size - len(batch))],
                                  length=self.memory_size)

        ids_lengths = np.array([len(item['ids']) for item in batch] + [len(batch[0]['ids'])
                               for _ in range(batch_size - len(batch))])

        labels = np.zeros((len(batch), 4), dtype=np.int32)  # [batch_size(32), label types(4)]
        for i, item in enumerate(batch):
            try:
                lab = [x for x in item['label_names']]
                # lab[1], lab[3] = lab[1].split(".")[-1], lab[3].split(".")[-1]
                lab = [self.label2index[i][lab[i]] for i in range(4)]
                labels[i, :] = lab
            except:
                print("error", i)

        feed_dict = tf_utils.filter_none_keys({mtest.ids: ids,
                                               mtest.ids_lengths: ids_lengths})

        (preds, ) = session.run([mtest.preds], feed_dict)
        preds = preds[:batch_size_original]
        labels = labels[:batch_size_original]
        correct = (preds == labels)

        if self.label_types == [0, 1, 2, 3]:
          correct = np.concatenate((correct,
                                    np.all(correct[:, [0, 2]], axis=1)[:, np.newaxis],
                                    # np.all(correct[:, [1, 3]], axis=1)[:, np.newaxis],
                                    np.all(correct[:, [0,1,2, 3]], axis=1)[:, np.newaxis]),
                                    axis=1)

        return correct, len(preds), preds


def main(args):
    if args.logdir is None:
        args.logdir = tempfile.mkdtemp(prefix='ifttt_')
        print(args.logdir)
    if args.clear and not (args.number_logdir or args.test_logdir):
        try:
            shutil.rmtree(args.logdir)
        except OSError:
            pass


    # testing modeling configure
    if args.test_logdir:
        print("Testing model with best parameters settings...")
        config = json.load(open(os.path.join(args.logdir, 'config.json')))
        stats = json.load(open(os.path.join(args.logdir, 'stats.json')))

        print("building MainLoop ...")
        ifttt_train = MainLoop(args, config)

        with tf.variable_scope('model', reuse=None, initializer=None):
            print('create model....')
            m = ifttt_train.create_model(is_training=False)

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

                batch_size = ifttt_train.optim_config['batch_size']
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
                data_dict = {'probs': dict(probs_by_section), 'labels': dict(labels_by_section)}
                pickle.dump(data_dict, fw, pickle.HIGHEST_PROTOCOL)

    # for training...
    if args.config:
        config = json.load(open(args.config, 'r'))
        print("configure: {}".format(config))
        tf_utils.mkdir_p(args.logdir)
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

        dict_file = config['dict_file']
        label2index, index2label = defaultdict(), defaultdict()
        basic = ['zero2index', 'one2index', 'two2index', 'three2index']
        for index, choice in enumerate(basic):
            with codecs.open(dict_file[choice], 'r', 'utf-8')as fr:
                lines = fr.read().split('\n')
                label2index[index] = defaultdict(int)
                index2label[index] = defaultdict(str)
                for line in lines:
                    splitted = line.strip().split(' ')
                    if len(splitted) == 2:
                        label2index[index][splitted[0]] = int(splitted[1])
                        index2label[index][int(splitted[1])] = splitted[0]

        label_embd = []
        if args.label_embd:
            with codecs.open(args.label_embd, 'rb') as fr:
                data_dict = pickle.load(fr)
                for i in range(4):
                    mat = np.zeros((len(label2index[i]), config['model']['embedding_size']*2))
                    print(mat.shape)
                    for x in label2index[i]:
                        # print("before mat: {}".format(mat[i]))
                        if x not in data_dict[i]:
                            print("{} not found.".format(x))
                            continue
                        # print("datadict[{}]: {}".format(x, data_dict[i][x]))
                        mat[i] = data_dict[i][x]
                        # print("after: {}".format(mat[i]))
                    label_embd.append(np.transpose(mat))

        mainloop = MainLoop(args, config, label2index, index2label)
        try:
            # the training epoch
            mainloop.run(mainloop.initializer, args.logdir, args.load_model, label_embd)
        except KeyboardInterrupt:
              pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--label-embd')
    parser.add_argument('--load-model')
    parser.add_argument('--config')
    parser.add_argument('--number-logdir', action='store_true')
    parser.add_argument('--test-logdir', action='store_true')
    parser.add_argument('--logdir')
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--log-device-placement', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print("args", args)
    main(args)
