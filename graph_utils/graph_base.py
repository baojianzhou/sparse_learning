# -*- coding: utf-8 -*-
__all__ = ['simu_graph', 'node_pre_rec_fm']

import numpy as np


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
