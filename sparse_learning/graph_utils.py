# -*- coding: utf-8 -*-
import numpy as np

__all__ = ['simu_graph', 'draw_graph', 'node_pre_rec_fm']


def simu_graph(num_nodes):
    edges, weights = [], []
    length = int(np.sqrt(num_nodes))
    width, index = length, 0
    for i in range(length):
        for j in range(width):
            if (index % length) != (length - 1):
                edges.append((index, index + 1))
                weights.append(1.0)
                if index + length < int(width * length):
                    edges.append((index, index + length))
                    weights.append(1.0)
            else:
                if index + length < int(width * length):
                    edges.append((index, index + length))
                    weights.append(1.0)
            index += 1
    edges = np.asarray(edges, dtype=int)
    weights = np.asarray(weights, dtype=np.float64)
    return edges, weights


def draw_graph(sub_graph, edges, length, width):
    import networkx as nx
    from pylab import rcParams
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import subplots_adjust
    subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    rcParams['figure.figsize'] = 14, 14

    G = nx.Graph()
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    pos = dict()
    index = 0
    for i in range(length):
        for j in range(width):
            G.add_node(index)
            pos[index] = (j, length - i)
            index += 1
    nx.draw_networkx_nodes(G, pos, node_size=100, nodelist=range(33 * 33), node_color='gray')
    nx.draw_networkx_nodes(G, pos, node_size=100, nodelist=sub_graph, node_color='b')
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=2)
    plt.axis('off')
    plt.show()


def node_pre_rec_fm(true_feature, pred_feature):
    true_feature, pred_feature = set(true_feature), set(pred_feature)
    pre, rec, fm = 0.0, 0.0, 0.0
    if len(pred_feature) != 0:
        pre = len(true_feature & pred_feature) / float(len(pred_feature))
    if len(true_feature) != 0:
        rec = len(true_feature & pred_feature) / float(len(true_feature))
    if (pre + rec) > 0.:
        fm = (2. * pre * rec) / (pre + rec)
    return pre, rec, fm
