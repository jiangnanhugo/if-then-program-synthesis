from tf_utils.pymdd.mdd import *
import random
import pickle
import json
import os


def get_idx(temp_list, nodename, is_income=True):
    for i in range(len(temp_list)):
        if is_income and temp_list[i].in_come == nodename:
            return i
        if (not is_income) and temp_list[i].out_go == nodename:
            return i
    return -1


class Node(object):
    def __init__(self, layer, state, in_come, out_go):
        self.layer = layer

        def get_hash():
            return random.getrandbits(40)
        self.id = get_hash()
        self.Type = "node"
        self.state = state
        self.in_come = in_come
        self.out_go = out_go

    def get_string(self):
        return {'state': self.state, 'id': self.id,
                'layer': self.layer, 'Type': self.Type,
                'in_come': self.in_come, 'out_go': self.out_go}

    def __str__(self):
        return '{state: ' + self.state + ', id:' + str(self.id) + ', layer:' + str(self.layer) + \
               ', Type:' + str(self.Type) + ", in_:" + self.in_come + ', out_:' + self.out_go + '}'


class Arc(object):
    def __init__(self, head_id, tail_id, name):
        self.Type = "arc"
        self.label = name
        self.head = head_id
        self.tail = tail_id

    def get_string(self):
        return {'label': self.label, 'head': self.head,
                'tail': self.tail, 'Type': self.Type}

    def __str__(self):
        return '{type: ' + self.Type + ', label:' + self.label + \
               ', head:' + str(self.head) + ', tail:' + str(self.tail) + '}'


def build_nodes_arcs(label_names):
    node_list = []
    # 0-th layer
    uid = 0
    source = Node(0, "s", 'none', 'any')
    sink = Node(5, "t", 'any', 't')
    uid += 1
    node_list.append([source, ])

    # middle layer
    for layer in range(4):
        temp_list = []
        for i in range(len(label_names)):
            uid += 1
            if layer == 3:
                    new_node = Node(layer + 1, "u_" + str(uid), label_names[i][layer], "t")
            else:
                new_node = Node(layer + 1, "u_" + str(uid), label_names[i][layer], label_names[i][layer + 1])
            # print(new_node)
            temp_list.append(new_node)
        node_list.append(temp_list)
        # print(len(node_list[layer]))

    node_list.append([sink, ])

    # build arc
    arc_list = []
    temp_list = []
    source = node_list[0][0]
    for i in range(len(label_names)):
        dest = node_list[1][i]
        new_arc = Arc(dest.id, source.id, dest.in_come)
        temp_list.append(new_arc)
    arc_list.append(temp_list)

    for layer in range(1, 4):
        temp_list = []
        for i in range(len(label_names)):
            source = node_list[layer][i]
            dest = node_list[layer + 1][i]
            if source.out_go != dest.in_come:
                raise MemoryError("the edge does not match")
            new_arc = Arc(dest.id, source.id, source.out_go)
            temp_list.append(new_arc)
        arc_list.append(temp_list)

    temp_list = []
    dest = node_list[5][0]
    for i in range(len(label_names)):
        source = node_list[4][i]
        new_arc = Arc(dest.id, source.id, "None")
        temp_list.append(new_arc)
    arc_list.append(temp_list)

    json_list = [{"name": "IFTTT", "Type": "name"}]
    for i in range(len(node_list)):
        for x in node_list[i]:
            json_list.append(x.get_string())

    for i in range(len(arc_list)):
        for x in arc_list[i]:
            json_list.append(x.get_string())
    return json_list


def build_path(label_names):
    node_list = []
    # 0-th layer
    uid = 0
    source = Node(0, "s", 'none', 'any')
    sink = Node(5, "t", 'any', 't')
    uid += 1
    node_list.append(source)

    # middle layer
    for layer in range(4):
        if layer == 3:
            new_node = Node(layer + 1, "u_" + str(uid), "layer_"+str(layer), "t")
        else:
            new_node = Node(layer + 1, "u_" + str(uid), "layer_"+str(layer), "layer_"+str(layer+1))
        node_list.append(new_node)
        uid += 1
    node_list.append(sink)

    # build arc
    arc_list = []
    for layer in range(4):
        temp_list = []
        visited = set()
        for i in range(len(label_names)):
            # print(len(node_list), layer+1)
            if label_names[i][layer] not in visited:
                new_arc = Arc(node_list[layer+1].id, node_list[layer].id, label_names[i][layer])
                visited.add(label_names[i][layer])
                temp_list.append(new_arc)
        arc_list.append(temp_list)
    arc_list.append([Arc(node_list[-1].id, node_list[-2].id, "None")])

    init=[]
    for x in node_list:
        init.append(x.get_string())

    for i in range(len(arc_list)):
        for x in arc_list[i]:
            init.append(x.get_string())

    json_content = {"mdd_name": "IFTTT", "init": init, "paths": label_names}

    return json_content


def get_vps(label_names, maxwidth=0, out_dir="./model/", huristic_func=None):
    mymdd = MDD()
    filename = out_dir+"IFTTT-w-"+str(maxwidth)+'.pkl'
    print(filename)
    if os.path.isfile(filename):
        print("load_from_json content")
        with open(filename, 'rb')as fr:
            mymdd = pickle.load(fr)
    else:
        print("load raw and dump to file")
        json_content = build_path(label_names)
        if huristic_func:
            mymdd.loadJSON_with_huristic_width(json_content, maxwidth, huristic_func)
        else:
            mymdd.loadJSON_with_random_width(json_content, maxwidth)
        with open(filename, 'wb')as fw:
            pickle.dump(mymdd, fw, pickle.HIGHEST_PROTOCOL)
        print("dump to the pickle")
        with open(filename, 'rb')as fr:
            mymdd = pickle.load(fr)

    return mymdd
