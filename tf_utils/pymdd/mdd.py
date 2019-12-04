from itertools import chain # used in various places
from collections import deque # used in prune_recursive
from json import dump, load # used in dumpJSON and loadJSON
from collections import defaultdict
import numpy as np
from .MDDArc import MDDArc
import copy
from .MDDNode import MDDNode, MDDNodeInfo


class MDD(object):
    """MDD represents a multivalued decision diagram (MDD).

    Args:
        name (str): name of MDD (default: 'mdd')
        nodes (List[Dict[MDDNode, MDDNodeInfo]]): nodes of MDD;
            if None (default), set to empty list
    """

    def __init__(self, name='mdd', nodes=None):
        """Construct a new 'MDD' object."""
        # 'nodes' is a list of dicts (one for each node layer),
        # and each dict stores the nodes in that layer;
        # each node is represented as a (MDDNode, MDDNodeInfo) key-value pair
        self.nodes = nodes
        self.name = name
        if self.nodes is None:
            self.nodes = []

    @property
    def numNodeLayers(self):
        """Number of node layers; equal to number of 'variables' + 1."""
        return len(self.nodes)

    @property
    def numArcLayers(self):
        """Number of arc layers; equal to number of 'variables'."""
        return len(self.nodes)-1

    @property
    def widthList(self):
        """Number of nodes in each layer"""
        return list(len(lyr) for lyr in self.nodes)

    @property
    def maxWidth(self):
        """Maximum number of nodes in a single node layer."""
        return max(len(lyr) for lyr in self.nodes)

    def __str__(self, showLong=False, showIncoming=False):
        """Return a (human-readable) string representation of the MDD.

        Args:
            showLong (bool): use more vertical space (default: False)
            showIncoming (bool): show incoming arcs (default: False)

        Returns:
            str: string representation of MDD
        """
        s = '== MDD (' + self.name + ', ' + str(self.numArcLayers) + ' layers) ==\n'
        if showLong:
            # Long form
            s += '# Nodes\n'
            for (j, lyr) in enumerate(self.nodes):
                s += 'L' + str(j) + ':\n'
                for v in lyr:
                    s += '\t' + str(v) + ': <'
                    s += 'in={' + ', '.join(str(a) for a in self.nodes[j][v].incoming) + '}, '
                    s += 'out={' + ', '.join(str(a) for a in self.nodes[j][v].outgoing) + '}'
                    s += '>\n'
            s += '# (Outgoing) Arcs\n'
            s += '\n'.join(str(a) for a in self.alloutgoingarcs())
            if showIncoming:
                s += '\n# (Incoming) Arcs\n'
                s += '\n'.join(str(a) for a in self.allincomingarcs())
        else:
            # Short form
            s += '# Nodes\n'
            for (j, lyr) in enumerate(self.nodes):
                s += 'L' + str(j) + ': '
                s += ', '.join(str(v) for v in self.allnodes_in_layer(j)) + '\n'
            s += '# (Outgoing) Arcs\n'
            s += ', '.join(str(a) for a in self.alloutgoingarcs())
            if showIncoming:
                s += '\n# (Incoming) Arcs\n'
                s += ', '.join(str(a) for a in self.allincomingarcs())
        return s

    def __repr__(self):
        return 'MDD(' + repr(self.name) + ', ' + repr(self.nodes) + ')'

    def _get_node_info(self, node):
        """Get 'MDDNodeInfo' corresponding to 'node'.

        Get the 'MDDNodeInfo' object corresponding to the 'MDDNode'
        object 'node'. Note this function can *not* be used to populate
        the underlying dictionary; it can only be used to reference
        the object.
        """
        return self.nodes[node.layer][node]

    def _add_arc(self, newarc):
        """Add an arc to the MDD, without sanity checks."""
        self._get_node_info(newarc.tail).outgoing.add(newarc)
        self._get_node_info(newarc.head).incoming.add(newarc)

    def _remove_arc(self, rmvarc):
        """Remove an arc from the MDD, without sanity checks."""
        self._get_node_info(rmvarc.tail).outgoing.remove(rmvarc)
        self._get_node_info(rmvarc.head).incoming.remove(rmvarc)

    def _add_node(self, newnode):
        """Add a node to the MDD, without sanity checks."""
        # NOTE: If an identical node already exists, its incoming and outgoing
        # arcs will be ERASED!!!
        self.nodes[newnode.layer][newnode] = MDDNodeInfo()

    def _remove_node(self, rmvnode):
        """Remove a node from the MDD, without sanity checks."""
        for arc in self._get_node_info(rmvnode).incoming:
            self._get_node_info(arc.tail).outgoing.remove(arc)
        for arc in self._get_node_info(rmvnode).outgoing:
            self._get_node_info(arc.head).incoming.remove(arc)
        del self.nodes[rmvnode.layer][rmvnode]

    def _remove_nodes(self, rmvnodes):
        """Remove a list of nodes from the MDD, without sanity checks."""
        for v in rmvnodes:
            self._remove_node(v)

    # Default inarcfun, outarcfun methods
    @staticmethod
    def _default_inarcfun(mgnode, inarc, lyr):
        return MDDArc(inarc.label, inarc.tail, mgnode)
    @staticmethod
    def _default_outarcfun(mgnode, outarc, lyr):
        return MDDArc(outarc.label, mgnode, outarc.head)

    #
    def _merge_nodes_internal(self, mnodes, mlayer, nodefun, inarcfun=None, outarcfun=None):
        """Merge specified nodes into a new node, with MDDNode/MDDArc functions.

        Merge specified nodes into new supernode and modify arcs appropriately.
        The difference between this function and _merge_nodes is that nodefun,
        inarcfun, and outarcfun directly return MDDNodes and MDDArcs (as opposed
        to returning the new node state and new arc weights).

        Args:
            mnodes (List[MDDNode]): nodes to be merged together
            mlayer (int): layer containing merged nodes
                NOTE: all nodes in mnodes must be in layer mlayer
            nodefun (Callable[[List[MDDNode], int], MDDNode]):
                nodefun(vlist, j) returns the node resulting from merging nodes
                in 'vlist' in layer 'j'
            inarcfun (Callable[[MDDNode, MDDArc, int], MDDArc]):
                inarcfun(mgnode, inarc, j) returns the arc (corresponding to
                'inarc') incoming to the new merged node 'mgnode' in layer 'j';
                if inarcfun is None (default), the original 'inarc' data is
                used unchanged
                NOTE: head of returned arc must be 'mgnode'
            outarcfun (Callable[[MDDNode, MDDArc, int], MDDArc]):
                outarcfun(mgnode, outarc, j) returns the arc (corresponding to
                'outarc') outgoing from the new merged node 'mgnode'
                in layer 'j';
                if outarcfun is None (default), the original 'outarc' data is
                used unchanged
                NOTE: tail of returned arc must be 'mgnode'

        Returns:
            MDDNode: new merged supernode

        Raises:
            ValueError: cannot merge < 2 nodes
        """
        # Basic check
        if len(mnodes) < 2:
            raise ValueError('Cannot merge < 2 nodes: %s' % str(mnodes))
        # Use default inarcfun/outarcfun if unspecified
        if inarcfun is None:
            inarcfun = self._default_inarcfun
        if outarcfun is None:
            outarcfun = self._default_outarcfun

        # Enumerate incoming/outgoing arcs
        for v in mnodes:
            if v not in self.nodes[mlayer]:
                raise RuntimeError("{} does not in layer {}".format(v, mlayer))
        # print([self.nodes[mlayer][v] for v in mnodes])
        mIncoming = set(chain.from_iterable(self.nodes[mlayer][v].incoming for v in mnodes))
        mOutgoing = set(chain.from_iterable(self.nodes[mlayer][v].outgoing for v in mnodes))

        # Create new supernode, and new incoming/outgoing arcs
        mNode = nodefun(mnodes, mlayer)
        newIncoming = [inarcfun(mNode, arc, mlayer) for arc in mIncoming]
        newOutgoing = [outarcfun(mNode, arc, mlayer) for arc in mOutgoing]
        # Delete merged nodes
        self._remove_nodes(mnodes)
        # Add supernode and its arcs to MDD
        self._add_node(mNode)
        for arc in newIncoming:
            self._add_arc(arc)
        for arc in newOutgoing:
            self._add_arc(arc)

        # Return new merged supernode
        return mNode

    # Default awfun method
    @staticmethod
    def _default_awfun(w, ns=None, nt=None, j=None):
        return w

    def _merge_nodes(self, mnodes, mlayer, nsfun, awinfun=None, awoutfun=None):
        """
        Merge specified nodes into new supernode and modify arcs appropriately.
        The difference between this function and _merge_nodes_internal is that
        nsfun, awinfun, and awoutfun return the new node state and new arc
        weights for the merged supernode (as opposed to directly returning
        MDDNodes and MDDArcs).

        Args:
            mnodes (List[MDDNode]): nodes to be merged together
            mlayer (int): layer containing merged nodes
            nsfun (Callable[[List[object], int], object]): nsfun(slist,j) returns the
                node state resulting from merging node states in 'slist'
                in layer 'j'
            awinfun (Callable[[float, object, object, int], float]):
                awinfun(w,os,ms,j) returns the adjusted weight of an arc with
                weight 'w', old head node state 'os', and new head node (i.e.,
                merged supernode in layer 'j') state 'ms';
                if awinfun is None (default), the original weight is used
            awoutfun (Callable[[float, object, object, int], float]):
                awoutfun(w,os,ms,j) returns the adjusted weight of an arc with
                weight 'w', old tail node state 'os', and new tail node (i.e.,
                merged supernode in layer'j') state 'ms';
                if awoutfun is None (default), the original weight is used

        Returns:
            MDDNode: new merged supernode
        """
        # Use default awfun if unspecified
        if awinfun is None:
            awinfun = self._default_awfun
        if awoutfun is None:
            awoutfun = self._default_awfun

        def nodefun(vlist, lyr):
            return MDDNode(mlayer, nsfun([v.state for v in vlist], lyr))

        def inarcfun(mgnode, inarc, lyr):
            return MDDArc(inarc.label, inarc.tail, mgnode)

        def outarcfun(mgnode, outarc, lyr):
            return MDDArc(outarc.label, mgnode, outarc.head)

        return self._merge_nodes_internal(mnodes, mlayer, nodefun, inarcfun, outarcfun)

    def _append_new_layer(self):
        """Append a new layer to the MDD."""
        self.nodes.append(dict())

    def _clear(self):
        """Reset the MDD."""
        self.nodes = []

    def allnodes(self):
        """Return all MDDNodes in the MDD."""
        return chain.from_iterable(l.keys() for l in self.nodes)

    def allnodeitems_in_layer(self, layer):
        """Return all (MDDNode, MDDNodeInfo) pairs in a particular layer."""
        return self.nodes[layer].items()

    def allnodes_in_layer(self, layer):
        """Return all MDDNodes in a particular layer."""
        return self.nodes[layer].keys()

    def alloutgoingarcs(self):
        """Return all outgoing arcs in the MDD."""
        return chain.from_iterable(ui.outgoing for j in range(self.numArcLayers) for ui in self.nodes[j].values())

    def allincomingarcs(self):
        """Return all incoming arcs in the MDD."""
        return chain.from_iterable(ui.incoming for j in range(self.numArcLayers) for ui in self.nodes[j+1].values())

    def add_arc(self, newarc):
        """Add an arc to the MDD (with sanity checks).

        Args:
            newarc (MDDArc): arc to be added

        Raises:
            RuntimeError: head/tail node of arc does not exist
            ValueError: head and tail nodes must be one layer apart
        """
        if not newarc.tail in self.allnodes_in_layer(newarc.tail.layer):
            raise RuntimeError('tail node of arc does not exist')
        if not newarc.head in self.allnodes_in_layer(newarc.head.layer):
            raise RuntimeError('head node of arc does not exist')
        if newarc.head.layer != newarc.tail.layer + 1:
            raise ValueError('head and tail must be one layer apart (%d != %d + 1)' % (newarc.head.layer, newarc.tail.layer))
        self._add_arc(newarc)

    def remove_arc(self, rmvarc):
        """Remove an arc from the MDD (with sanity checks).

        Args:
            rmvarc (MDDArc): arc to be removed

        Raises:
            RuntimeError: head/tail node of arc does not exist
            KeyError: no such incoming/outgoing arc exists in the MDD
        """
        if not rmvarc.tail in self.allnodes_in_layer(rmvarc.tail.layer):
            raise RuntimeError('tail node of arc does not exist')
        if not rmvarc.head in self.allnodes_in_layer(rmvarc.head.layer):
            raise RuntimeError('head node of arc does not exist')
        if not rmvarc in self._get_node_info(rmvarc.tail).outgoing:
            raise KeyError('cannot remove non-existent outgoing arc')
        if not rmvarc in self._get_node_info(rmvarc.head).incoming:
            raise KeyError('cannot remove non-existent incoming arc')
        self._remove_arc(rmvarc)

    def add_node(self, newnode):
        """Add a new node to the MDD (with sanity checks).

        Args:
            newnode (MDDNode): node to be added

        Raises:
            IndexError: the MDD does not contain the specified node layer
            ValueError: a duplicate node already exists in the MDD
        """
        if newnode.layer >= self.numNodeLayers or newnode.layer < 0:
            raise IndexError('node layer %d does not exist' % newnode.layer)
        if newnode in self.allnodes_in_layer(newnode.layer):
            raise ValueError('cannot add proposed node; duplicate node already exists')
        self._add_node(newnode)

    def remove_node(self, rmvnode):
        """Remove a node from the MDD (with sanity checks).

        Args:
            rmvnode (MDDNode): node to be removed

        Raises:
            IndexError: the MDD does not contain the specified node layer
            KeyError: no such node exists in the MDD
        """
        if rmvnode.layer >= self.numNodeLayers or rmvnode.layer < 0:
            raise IndexError('node layer %d does not exist' % rmvnode.layer)
        if not rmvnode in self.allnodes_in_layer(rmvnode.layer):
            raise KeyError('cannot remove non-existent node')
        self._remove_node(rmvnode)

    def merge_nodes(self, mnodes, nsfun, awinfun=None, awoutfun=None):
        """Merge nodes in mnodes into a new supernode.  The state of the new
        merged node is specified by nsfun, while the weights of incoming
        and/or outgoing arcs are specified by awinfun/awoutfun.

        Args:
            mnodes (List[MDDNode]): nodes to be merged together
            nsfun (Callable[[List[object], int], object]):
                nsfun(slist, j) returns the node state resulting from merging
                node states in 'slist' in layer 'j'
            awinfun (Callable[[float, object, object, int], float]):
                awinfun(w,os,ms,j) returns the adjusted weight of an arc with
                weight 'w', old head node state 'os', and new head node (i.e.,
                merged supernode in layer 'j') state 'ms';
                if awinfun is None (default), the original weight is used
            awoutfun (Callable[[float, object, object, int], float]):
                awoutfun(w,os,ms,j) returns the adjusted weight of an arc with
                weight 'w', old tail node state 'os', and new tail node (i.e.,
                merged supernode in layer 'j') state 'ms';
                if awoutfun is None (default), the original weight is used

        Returns:
            MDDNode: new merged supernode

        Raises:
            ValueError: cannot merge nodes in different layers
        """
        # Check all nodes in mnodes are on same layer
        mlayer = [v.layer for v in mnodes]
        if len(set(mlayer)) > 1:
            raise ValueError('cannot merge nodes in different layers')
        return self._merge_nodes(mnodes, mlayer[0], nsfun, awinfun, awoutfun)



    def remove_redundant_arc_within_states(self, mergeFunc, adjInFunc=None, adjOutFunc=None):
        j=0
        while j <= self.numArcLayers:
            inDict = defaultdict(list)
            for v in self.allnodes_in_layer(j):
                for a in self.nodes[j][v].incoming:
                    inNeighbor = (a.label, a.head)
                    inDict[str(a.tail)+ a.label].append(inNeighbor)

            has_dup=False

            for key in inDict:
                mnodes=inDict[key]
                if len(mnodes) >= 2:
                    redundant_nodes = [x[1] for x in mnodes]
                    # print("remove redundant nodes {} :{}".format(key, redundant_nodes))
                    self._merge_nodes(redundant_nodes, j, mergeFunc, adjInFunc, adjOutFunc)
                    has_dup = True
                    break
            if len(inDict)==0 or has_dup == False:
                j += 1

    def find_neighbor_along_path(self, arc_names):
        cur_state = self.allnodes_in_layer(0)
        cur_state = list(cur_state)[0]
        neighbors_labels=[]
        for j, name in enumerate(arc_names):
            next_state, neighbors=self._find_next_state(cur_state, name)
            cur_state = next_state
            neighbors_labels.append(neighbors)

        neighbors_labels.append(self.get_outgoing_of_state_at_layer(cur_state, len(arc_names)))
        return neighbors_labels

    def get_outgoing_of_state_at_layer(self, cur_state, layer):
        for (u, ui) in self.allnodeitems_in_layer(layer):
            if u != cur_state:
                continue
            label_valid_assignment = [a.label for a in ui.outgoing]
            return label_valid_assignment

    def find_last_neighbor_along_path(self, arc_names):
        cur_state = self.allnodes_in_layer(0)
        cur_state = list(cur_state)[0]
        neighbors_labels=[]
        for j, name in enumerate(arc_names):
            next_state, neighbors = self._find_next_state(cur_state, name)
            cur_state = next_state
            neighbors_labels.append([name])
        neighbors_labels.append(self.get_outgoing_of_state_at_layer(cur_state, len(arc_names)))
        return neighbors_labels

    def _find_next_state(self, cur_state, name):
        all_out_arcs=[x for x in self.nodes[cur_state.layer][cur_state].outgoing]
        neighbors = [x.label for x in all_out_arcs]
        for out_arc in all_out_arcs:
            if out_arc.label == name:
                return out_arc.head, neighbors

    # Default functions/args for GraphViz output
    @staticmethod
    def _default_ndf(state, layer):
        return 'label="%s"' % str(state)

    @staticmethod
    def _default_adf(label, layer):
        return 'label="%s"' % label

    _default_asa = {'key': lambda a: a.label}
    _default_nsa = {'key': lambda v: v.state, 'reverse': True}

    def output_to_dot(self, nodeDotFunc=None, arcDotFunc=None, arcSortArgs=None, nodeSortArgs=None, reverseDir=False, fname=None):
        """Write the graphical structure of the MDD to a file.

        Write the graphical structure of the MDD to a file (<MDDName>.gv) in
        the DOT language.  The MDD can then be visualized with GraphViz.

        Args:
            nodeDotFunc (Callable[[object, int], str]): nodeDotFunc(s,j)
                returns a string with the DOT options to use given node state
                's' in layer 'j'; if None (default), a sensible default is used
            arcDotFunc (Callable[[object, float, int], str]): arcDotFunc(l,w,j)
                returns a string with the DOT options to use given arc label
                'l', arc weight 'w', and tail node layer 'j'; if None (default),
                a sensible default is used
            arcSortArgs (dict): arguments specifying how to sort a list of arcs
                via list.sort() (i.e., 'key' and, optionally, 'reverse');
                GraphViz then attempts to order the arcs accordingly in the
                output graph; if arcSortArgs is None (default), no such order
                is enforced
            nodeSortArgs (dict): arguments specifying how to sort a list of
                nodes via list.sort() (i.e., 'key' and, optionally, 'reverse');
                GraphViz then attempts to order the nodes accordingly in the
                output graph; if nodeSortArgs is None (default), no such order
                is enforced
            reverseDir (bool): if True, show the MDD with arcs oriented in the
                opposite direction, so the terminal node appears at the top and
                the root node at the bottom (default: False)
            fname (str): name of output file; if None, default to <MDDName>.gv
        """

        # Use default output functions if unspecified
        if nodeDotFunc is None:
            nodeDotFunc = self._default_ndf
        if arcDotFunc is None:
            arcDotFunc = self._default_adf
        if reverseDir:
            iterRange = range(self.numArcLayers, 0, -1)
            (nextArcAttr, srcAttr, destAttr) = ('incoming', 'head', 'tail')
        else:
            iterRange = range(self.numArcLayers)
            (nextArcAttr, srcAttr, destAttr) = ('outgoing', 'tail', 'head')
        if fname is None:
            fname = '%s.gv' % self.name

        outf = open(fname, 'w')
        outf.write('digraph "%s" {\n' % self.name)
        outf.write('graph[fontname="Monospace Regular"];\nnode[fontname="Monospace Regular"];\nedge[fontname="Monospace Regular"];\n')
        if reverseDir:
            outf.write('edge [dir=back];\n')
        if arcSortArgs is not None:
            outf.write('ordering=out;\n')
        for v in self.allnodes():
            outf.write('%d[%s];\n' % (hash(v), nodeDotFunc(v.state, v.layer)))
        for j in iterRange:
            for (u, ui) in self.allnodeitems_in_layer(j):
                arcsinlayer = [a for a in getattr(ui, nextArcAttr)]
                if arcSortArgs is not None:
                    arcsinlayer.sort(**arcSortArgs)
                for arc in arcsinlayer:
                    outf.write('%d -> %d[%s];\n' % (hash(getattr(arc, srcAttr)), hash(getattr(arc, destAttr)), arcDotFunc(arc.label, arc.tail.layer)))
        if nodeSortArgs is not None:
            for j in range(self.numNodeLayers):
                nodesinlayer = [v for v in self.allnodes_in_layer(j)]
                if len(nodesinlayer) > 1:
                    nodesinlayer.sort(**nodeSortArgs)
                    for i in range(len(nodesinlayer) - 1):
                        outf.write('%d -> %d[style=invis];\n' % (hash(nodesinlayer[i]), hash(nodesinlayer[i+1])))
                    outf.write('{rank=same')
                    for v in nodesinlayer:
                        outf.write(';%d' % hash(v))
                    outf.write('}\n')
        outf.write('}')
        outf.close()

    def dumpJSON(self, fname=None, stateDumpFunc=repr, labelDumpFunc=repr):
        """Dump the MDD into a JSON file.

        Dump the contents of the MDD into a JSON file for later retrieval.

        Args:
            fname (str): name of json file (default: self.name + '.json')
            stateDumpFunc (Callable[[object], str]): stateDumpFunc(s) returns
                a string representation of the node state 's' (default: repr)
            labelDumpFunc (Callable[[object], str]): labelDumpFunc(l) returns
                a string representation of the arc label 'l' (default: repr)
        """
        if fname is None:
            fname = self.name + '.json'
        dataList = []
        dataList.append({'Type': 'name', 'name': self.name})
        for v in self.allnodes():
            dataList.append({'Type': 'node', 'layer': v.layer, 'state': stateDumpFunc(v.state), 'id': hash(v)})
        for a in self.alloutgoingarcs():
            dataList.append({'Type': 'arc', 'label': labelDumpFunc(a.label), 'tail': hash(a.tail), 'head': hash(a.head)})
        outf = open(fname, 'w')
        dump(dataList, outf)
        outf.close()

    def loadJSON_with_random_width(self, json_content, maxWidth):
        """Load an MDD from a JSON file."""
        self._clear()

        dataList = json_content

        nodeDict = dict()
        self.raw_input = dataList['paths']
        self.name = dataList['mdd_name']
        for item in dataList['init']:
            if item['Type'] == 'node':
                while int(item['layer']) >= self.numNodeLayers:
                    self._append_new_layer()
                newnode = MDDNode(int(item['layer']), item['state'])
                self.add_node(newnode)
                nodeDict[item['id']] = newnode
            elif item['Type'] == 'arc':
                newarc = MDDArc(item['label'], nodeDict[item['tail']], nodeDict[item['head']])
                self.add_arc(newarc)
            else:
                raise ValueError('Unknown item type: check input file format')

        j = 1
        node_idx = 6
        print("numArcLayers: {}".format(self.numArcLayers))
        while j < self.numArcLayers and len(self.allnodes_in_layer(j)) < maxWidth:
            print(j, len(self.allnodes_in_layer(j)), maxWidth)
            nodesinlayer = [v for v in self.allnodes_in_layer(j)]
            for v in nodesinlayer:
                length = len(self.nodes[j][v].incoming)
                print("layer {} incoming edges: {}".format(j, length))

                rand_bits = np.random.randint(2, size=length)
                if np.count_nonzero(rand_bits) == 0 or np.count_nonzero(rand_bits==0)==0:
                    print("one of splitted has no income edge:{} {}".format(np.count_nonzero(rand_bits),
                                                                            np.count_nonzero(rand_bits==0)))
                    continue
                if len(self.nodes[j]) > maxWidth - 1:
                    print("node canot be splitted, beacause reaching the bound!:", len(self.nodes[j]))
                    continue
                new_nodes1 = MDDNode(layer=j, state="u_"+str(node_idx))
                new_nodes1_income = set()
                new_nodes2 = MDDNode(layer=j, state="u_"+str(node_idx+1))
                new_nodes2_income = set()

                self.add_node(new_nodes1)
                self.add_node(new_nodes2)
                node_idx += 2
                for i, a in enumerate(self.nodes[j][v].incoming):       #set()
                        if rand_bits[i] == 1:
                            newarc = MDDArc(a.label, a.tail, new_nodes1)
                            self.add_arc(newarc)
                            new_nodes1_income.add(a.label)
                        elif rand_bits[i] == 0:
                            newarc = MDDArc(a.label, a.tail, new_nodes2)
                            self.add_arc(newarc)
                            new_nodes2_income.add(a.label)
                print("split the incoming edges:", len(self.nodes[j][v].incoming), len(self.nodes[j][new_nodes1].incoming), len(self.nodes[j][new_nodes2].incoming))
                tupled=defaultdict(set)
                for i in range(len(self.raw_input)):
                        element = self.raw_input[i][j-1]
                        if j < len(self.raw_input[i]):
                            next_element = self.raw_input[i][j]
                        else:
                            next_element = "None"
                        tupled[element].add(next_element)
                print("the value pair of raw_data is: {}".format(np.sum([len(tupled[t]) for t in tupled])))
                for income_arc in tupled:
                    if income_arc in new_nodes1_income:
                        for out_arc in tupled[income_arc]:
                            picked_arc=None
                            for x in self.nodes[j][v].outgoing:
                                if x.label == out_arc:
                                    picked_arc = x
                                    break
                            newarc1 = MDDArc(out_arc, 1.0, new_nodes1, picked_arc.head)
                            self.add_arc(newarc1)
                for income_arc in tupled:
                    if income_arc in new_nodes2_income:
                        for out_arc in tupled[income_arc]:
                            picked_arc=None
                            for x in self.nodes[j][v].outgoing:
                                if x.label == out_arc:
                                    picked_arc = x
                                    break
                            newarc2 = MDDArc(out_arc, 1.0, new_nodes2, picked_arc.head)
                            self.add_arc(newarc2)
                print("split the outgoings edges:", len(self.nodes[j][v].outgoing),
                      len(self.nodes[j][new_nodes1].outgoing),
                      len(self.nodes[j][new_nodes2].outgoing),
                      len(self.nodes[j][new_nodes1].outgoing)+ len(self.nodes[j][new_nodes2].outgoing))
                print("add node: {} {}, remove node: {}".format(new_nodes1, new_nodes2, v))
                self.remove_node(v)
            j += 1
            if j >= self.numArcLayers:
                j = 1
            # break
    @staticmethod
    def huristic_generate_node(category, income_edges):
        grouped_in_edges=defaultdict(set)
        for edge in income_edges:
            for ca in category:
                if edge.label in category[ca]:
                    grouped_in_edges[ca].add(edge)
        max_idx=None
        for x in grouped_in_edges:
            if max_idx == None:
                max_idx=x
            if len(grouped_in_edges[x]) >= len(grouped_in_edges[max_idx]):
                max_idx=x
        return grouped_in_edges[max_idx]

    def loadJSON_with_huristic_width(self, json_content, maxWidth, cat_huristic=None):
        """Load an MDD from a JSON file."""
        self._clear()

        dataList = json_content

        nodeDict = dict()
        self.raw_input = dataList['paths']
        self.name = dataList['mdd_name']
        for item in dataList['init']:
            if item['Type'] == 'node':
                while int(item['layer']) >= self.numNodeLayers:
                    self._append_new_layer()
                newnode = MDDNode(int(item['layer']), item['state'])
                self.add_node(newnode)
                nodeDict[item['id']] = newnode
            elif item['Type'] == 'arc':
                newarc = MDDArc(item['label'], nodeDict[item['tail']], nodeDict[item['head']])
                self.add_arc(newarc)
            else:
                raise ValueError('Unknown item type: check input file format')

        j = 1
        node_idx = 6
        print("numArcLayers: {}".format(self.numArcLayers))
        while j < self.numArcLayers and len(self.allnodes_in_layer(j)) < maxWidth:
            print(j, len(self.allnodes_in_layer(j)), maxWidth)
            # find a node in layer_j with largest income edges
            max_idx = None
            for v in self.allnodes_in_layer(j):
                if max_idx == None:
                    max_idx = v
                if len(self.nodes[j][v].incoming) > len(self.nodes[j][max_idx].incoming):
                    max_idx=v
            chosen_nodeinfo=self.nodes[j][max_idx]
            length = len(chosen_nodeinfo.incoming)
            print("layer {} incoming edges: {}".format(j, length))
            one_in_arcs = self.huristic_generate_node(cat_huristic, chosen_nodeinfo.incoming)
            if len(one_in_arcs) == 0 or len(one_in_arcs) == length:
                    print("cannot split the node into two: {} {}".format(len(one_in_arcs), length))
                    continue
            if len(self.nodes[j]) > maxWidth - 1:
                    print("node canot be splitted, beacause reaching the bound!:", len(self.nodes[j]))
                    continue
            new_nodes1 = MDDNode(layer=j, state="u_"+str(node_idx))
            new_nodes1_income = set()
            new_nodes2 = MDDNode(layer=j, state="u_"+str(node_idx+1))
            new_nodes2_income = set()

            self.add_node(new_nodes1)
            self.add_node(new_nodes2)
            node_idx += 2
            for a in chosen_nodeinfo.incoming:       #set()
                    if a in one_in_arcs:
                        newarc = MDDArc(a.label, a.tail, new_nodes1)
                        self.add_arc(newarc)
                        new_nodes1_income.add(a.label)
                    else:
                        newarc = MDDArc(a.label, a.tail, new_nodes2)
                        self.add_arc(newarc)
                        new_nodes2_income.add(a.label)
            print("split the incoming edges:", len(chosen_nodeinfo.incoming), len(self.nodes[j][new_nodes1].incoming), len(self.nodes[j][new_nodes2].incoming))
            tupled=defaultdict(set)
            for i in range(len(self.raw_input)):
                        element = self.raw_input[i][j-1]
                        if j < len(self.raw_input[i]):
                            next_element = self.raw_input[i][j]
                        else:
                            next_element = "None"
                        tupled[element].add(next_element)
            print("the value pair of raw_data is: {}".format(np.sum([len(tupled[t]) for t in tupled])))
            for income_arc in tupled:
                    if income_arc in new_nodes1_income:
                        for out_arc in tupled[income_arc]:
                            picked_arc=None
                            for x in chosen_nodeinfo.outgoing:
                                if x.label == out_arc:
                                    picked_arc = x
                                    break
                            newarc1 = MDDArc(out_arc, new_nodes1, picked_arc.head)
                            self.add_arc(newarc1)
            for income_arc in tupled:
                    if income_arc in new_nodes2_income:
                        for out_arc in tupled[income_arc]:
                            picked_arc=None
                            for x in chosen_nodeinfo.outgoing:
                                if x.label == out_arc:
                                    picked_arc = x
                                    break
                            newarc2 = MDDArc(out_arc, new_nodes2, picked_arc.head)
                            self.add_arc(newarc2)
            print("split the outgoings edges:", len(chosen_nodeinfo.outgoing),
                      len(self.nodes[j][new_nodes1].outgoing),
                      len(self.nodes[j][new_nodes2].outgoing),
                      len(self.nodes[j][new_nodes1].outgoing)+ len(self.nodes[j][new_nodes2].outgoing))
            print("add node: {} {}, remove node: {}".format(new_nodes1, new_nodes2, chosen_nodeinfo))
            self.remove_node(max_idx)
            j += 1
            if j >= self.numArcLayers:
                j = 1
            # break
