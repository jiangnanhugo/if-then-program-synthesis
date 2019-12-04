import pickle
import json
from copy import deepcopy
import codecs
import numpy as np
from collections import defaultdict
import os
import pickle
import shutil

filename="msr_data.pkl"
with open(filename, 'rb') as f:
    datadict = pickle.load(f, encoding='latin1')

label2index=[defaultdict(),defaultdict(),defaultdict(),defaultdict()]
idx=[0,0,0,0]


for x in datadict['train']+datadict['dev']:
    na=[it.lower() for it in x['label_names']]
    #na[1]=".".join([na[0],na[1]])
    #na[3]=".".join([na[2],na[3]])
    x['label_names']=na
    for i in range(len(na)):
        if na[i] not in label2index[i]:
            label2index[i][na[i]]=idx[i]
            idx[i]+=1


index2label=[]
for i,x in enumerate(label2index):
    temp=defaultdict()
    for k,v in x.items():
        temp[v]=k
    index2label.append(temp)

datadict['index2label']=index2label
datadict['label2index']=label2index
print(idx)

      # os.makedirs(basepath,exist_ok=True)
with codecs.open("msr_data.pkl",'wb')as fw:
    pickle.dump(datadict,fw)
"""
for x in datadict['test']:
    na=[it.lower() for it in x['label_names']]
    #na[1]=".".join([na[0],na[1]])
    #na[3]=".".join([na[2],na[3]])


print(datadict['train'][0])
index2label=[]
for i,x in enumerate(label2index):
    temp=defaultdict()
    for k,v in x.items():
        temp[v]=k
   index2label.append(temp)
datadict['index2label']=index2label
datadict['label2index']=label2index
print(idx)

basepath='ext_'
# os.makedirs(basepath,exist_ok=True)
with codecs.open(basepath+"msr_data.pkl",'wb')as fw:
    pickle.dump(datadict,fw)

zero_one=np.zeros((idx[0],idx[1]),dtype=np.int32)
for x in datadict['train']+datadict['dev']:
    na=x['label_names']
    x=int(label2index[0][na[0]])
    y=int(label2index[1][na[1]])
    zero_one[x][y]=1


with codecs.open(basepath+"zero_one_bipartite.txt",'w','utf-8')as fw:
    fw.write("{} {} {}\n".format(idx[0], max(np.sum(zero_one, axis=1)), idx[1]))
    for i in range(len(label2index[0])):
        for j in range(len(label2index[1])):
            if zero_one[i,j]!=0.0:
                fw.write("{} {}\n".format(i,j))
    fw.flush()
    fw.close()

one_two=np.zeros((idx[1],idx[2]),dtype=np.int32)
for x in datadict['train']+datadict['dev']:
    na=x['label_names']
    x=int(label2index[1][na[1]])
    y=int(label2index[2][na[2]])
    one_two[x][y]=1

with codecs.open(basepath+"one_two_bipartite.txt",'w','utf-8')as fw:
    fw.write("{} {} {}\n".format(idx[1], max(np.sum(one_two, axis=1)), idx[2]))
    for i in range(len(label2index[1])):
        for j in range(len(label2index[2])):
            if one_two[i,j]!=0.0:
                fw.write("{} {}\n".format(i,j))
    fw.flush()
    fw.close()

# label[2] -> label[3]
two_three=np.zeros((idx[2],idx[3]),dtype=np.int32)
for x in datadict['train']+datadict['dev']:
    na=x['label_names']
    x=int(label2index[2][na[2]])
    y=int(label2index[3][na[3]])
    two_three[x][y]=1

with codecs.open(basepath+"two_three_bipartite.txt",'w','utf-8')as fw:
    fw.write("{} {} {}\n".format(idx[2], max(np.sum(two_three, axis=1)), idx[3]))
    for i in range(len(label2index[2])):
        for j in range(len(label2index[3])):
            if two_three[i,j]!=0.0:
                fw.write("{} {}\n".format(i,j))
    fw.flush()
    fw.close()"""

