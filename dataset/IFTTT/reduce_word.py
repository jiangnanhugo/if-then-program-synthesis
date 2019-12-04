import numpy as np
import os
import sys
import codecs
import pickle

def reduce_word_embed(data_path, embd_path):
    data = pickle.load(open(data_path, 'rb'), encoding="latin1")
    wordset = set()
    for item in (data['train'] + data['dev'] + data['test']):
        for w in item['words']:
            wordset.add(w)
    worddict = dict()
    with codecs.open(embd_path, 'r', 'utf-8') as fr:
        line = fr.readline()
        line = fr.readline().strip()
        while line:
            line = line.split(' ')
            if line[0] in wordset:
                worddict[line[0]]=line[1:]
            line = fr.readline().strip()
    fw = codecs.open(embd_path+'.reduced', 'w', 'utf-8')
    for w in worddict:
        fw.write(w+' ')
        fw.write(" ".join(worddict[w]))
        fw.write('\n')
    fw.flush()
    fw.close()


if __name__ == "__main__":
    reduce_word_embed('msr_data.pkl', '../wiki-news-300d-1M.vec')