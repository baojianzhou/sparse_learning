# -*- coding: utf-8 -*-
import os
import sys
import time
import pickle
import numpy as np
import multiprocessing
import pandas as pd
from itertools import product
from pcst_fast import pcst_fast
from online_graph_ghtp import online_graph_ghtp as c_gghpt


def online_gghtp_algo(x, y, w0, edges, costs, pathway_id, s_low, s_high, edge_costs_multiplier, root,
                      num_active_clusters, pruning, verbosity_level, max_num_iter):
    if x.dtype != np.float64 or y.dtype != np.float64 or edges.dtype != int or costs.dtype != np.float64:
        print('data type error. should be np.float64')
        exit()
    wt, run_times, losses = c_gghpt(x, y, w0, edges, costs, pathway_id, s_low, s_high, edge_costs_multiplier, root,
                                    num_active_clusters, pruning, verbosity_level, max_num_iter)

    return wt, run_times, losses


def run_single(paras):
    num_tasks, task_i, pathway_id, pathways, file_path, (edges, costs, nodes), s_low, s_high = paras
    dataset = pd.read_csv(file_path + '_inbiomap_exp.csv', index_col=0)
    dataset = dataset.transpose().reindex(index=nodes).transpose()
    edges, costs = np.asarray(edges, dtype=int), np.asarray(costs, dtype=np.float64)
    x, labels = np.asarray(dataset.values, dtype=np.float64), dataset.index.tolist()
    y, p = np.ones(len(x), dtype=np.float64), len(x[0])

    """
    selected_index, unselected_index = set(), set()
    for ind, node in enumerate(nodes):
        if node in pathways[pathway_id]:
            selected_index.add(ind)
        else:
            unselected_index.add(ind)
    x_ = np.zeros_like(x)
    x_[:, 0:len(list(selected_index))] = x[:, list(selected_index)]
    x_[:, len(list(selected_index)):] = x[:, list(unselected_index)]
    import matplotlib.pyplot as plt
    plt.imshow(x_)
    plt.show()
    exit()
    """
    for i in range(len(labels)):
        if labels[i] == 'negative':
            y[i] = -1  # using {+1,-1} label
    re = online_gghtp_algo(x=x, y=y, w0=np.zeros(p), edges=edges, costs=costs, pathway_id=pathway_id, s_low=s_low,
                           s_high=s_high, edge_costs_multiplier=2, root=-1, num_active_clusters=1, pruning='strong',
                           verbosity_level=0, max_num_iter=20)
    wt, run_times, losses = re
    print(wt)
    print(run_times)
    print(losses)


def main():
    if sys.argv[1] == 's1':
        data_path = input_data_path + 'kegg_500/strategy_1_'
        edges, costs, nodes, pathways = pickle.load(open(input_data_path + 'edges_costs_nodes_pathways.pkl'))
        sparsity_list = zip(np.linspace(0, 300, 16, dtype=int), np.linspace(20, 320, 16, dtype=int))  # 16 parameters
        num_tasks, graph, input_paras = len(sparsity_list) * len(pathways.keys()), (edges, costs, nodes), []
        for task_i, (path_id, (low, high)) in enumerate(product(pathways.keys(), sparsity_list)):
            input_paras.append((num_tasks, task_i, path_id, pathways, data_path + path_id, graph, low, high))
        pool = multiprocessing.Pool(processes=int(sys.argv[2]))
        run_single(input_paras[0])
        exit()
        results = pool.map(run_single, input_paras)
        pickle.dump(results, open(output_path + 'kegg_500_online_graph_ghtp_strategy_1.pkl', 'wb'))
    elif sys.argv[1] == 's2':
        data_path = input_data_path + 'kegg_500/strategy_2_'
        edges, costs, nodes, pathways = pickle.load(open(input_data_path + 'edges_costs_nodes_pathways.pkl'))
        sparsity_list = zip(np.linspace(0, 300, 16, dtype=int), np.linspace(20, 320, 16, dtype=int))  # 16 parameters
        num_tasks, graph, input_paras = len(sparsity_list) * len(pathways.keys()), (edges, costs, nodes), []
        for task_i, (path_id, (low, high)) in enumerate(product(pathways.keys(), sparsity_list)):
            input_paras.append((num_tasks, task_i, path_id, pathways, data_path + path_id, graph, low, high))
        pool = multiprocessing.Pool(processes=int(sys.argv[2]))
        results = pool.map(run_single, input_paras)
        pickle.dump(results, open(output_path + 'kegg_500_online_graph_ghtp_strategy_2.pkl', 'wb'))
    else:
        print('unknown methods ')
        exit()


if __name__ == "__main__":
    input_data_path = '/network/rit/lab/ceashpc/bz383376/data/icdm18/kegg/'
    output_path = '/network/rit/lab/ceashpc/bz383376/data/icdm18/kegg/output/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    main()
