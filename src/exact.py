import torch
import numpy as np
import networkx as nx
from utils import readgraph, sum_up_tree, energy_ising


def exact_config(D):
    config = np.zeros([2**D, D], dtype=np.float32)
    for i in range(2**D-1, -1, -1):
        num = i
        for j in range(D-1, -1, -1):
            config[i, D-j-1] = num // 2**j
            if num - 2**j >= 0:
                num -= 2**j
    return config


def exact_Z(D, beta, device):
    num_edges, edges, _, J = readgraph(D)

    J = torch.from_numpy(J).to(device)

    G = nx.Graph()
    G.add_nodes_from(np.arange(D))
    G.add_edges_from(edges)

    FVS = np.loadtxt('../data/fvs{}nodes.txt'.format(D)).astype(np.int)
    FVS = FVS.tolist()
    with open('../data/trees{}nodes.txt'.format(D)) as f:
        list1 = f.readlines()
    f.close()
    tree1 = []
    tree_hierarchy = []
    for i in range(len(list1)):
        current_line = list(map(int, list1[i].split()))
        tree1 += current_line
        tree_hierarchy.append(current_line)

    l = len(FVS)
    sample = torch.from_numpy(exact_config(l) * 2.0 - 1.0).to(device)
    fe_tree = sum_up_tree(sample, J, FVS, tree1, tree_hierarchy,
                          sample.shape[0], beta, device)
    energy = energy_ising(sample, J[FVS][:, FVS], l) + fe_tree
    partition_function = torch.sum(torch.exp(-beta * energy))

    return partition_function


if __name__ == '__main__':
    D = 60
    beta = 1
    device = 'cpu'
    Z = exact_Z(D, beta, device)
    print(Z/D)
