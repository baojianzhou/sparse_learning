# -*- coding: utf-8 -*-
__all__ = ['simu_graph']

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
