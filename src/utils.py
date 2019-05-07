import numpy as np
import torch
from collections import namedtuple

def get_args(method, nd, nw, beta, device='cpu'):
    args = namedtuple('args', ['dtype', 'D', 'method', 'device', 'num_epochs',
        'sample', 'calc', 'beta', 'net_depth', 'net_width'])

    return args('float64', 60, method, device, 5000, 10000, 100000, beta, nd, nw)

def readgraph(D):
    with open('../data/{}nodes.txt'.format(D), 'r') as f:
        list1 = f.readlines()
    f.close()
    num_edges = int(list1[0].split()[1])
    edges = np.zeros([len(list1)-1, 2], dtype=int)
    for i in range(len(list1)-1):
        edges[i] = list1[i+1].split()
    neighbors = {}.fromkeys(np.arange(D))
    for key in neighbors.keys():
        neighbors[key] = []
    for edge in edges:
        neighbors[edge[0]].append(edge[1])
        neighbors[edge[1]].append(edge[0])

    J = np.loadtxt('../data/energy_function{}nodes.txt'.format(D), dtype=np.float32)

    return num_edges, edges, neighbors, J


def energy_ising(sample, J, D):
    batch = sample.shape[0]
    J = J.to_sparse()
    energy = -torch.bmm(sample.view(batch, 1, D),
            torch.sparse.mm(J, sample.t()).t().view(batch, D, 1)).reshape(batch) / 2

    return energy


def sum_up_tree(sample, J, frozen_set, tree1, tree_hierarchy, sample_size, beta,
        device='cpu'):
    h = sample.matmul(J[frozen_set, :])
    fe_tree = torch.zeros(sample_size, device=device, dtype=sample.dtype)
    tree = torch.from_numpy(np.array(tree1)).to(device)
    for layer in tree_hierarchy:
        index_matrix = torch.zeros(len(layer), 2, dtype=torch.int64,
                                   device=device)
        index_matrix[:, 0] = torch.arange(len(layer))
        if len(J[layer][:, tree].nonzero()) != 0:
            index_matrix.index_copy_(0,
                                     J[layer][:, tree].nonzero()[:, 0],
                                     J[layer][:, tree].nonzero())
        index = index_matrix[:, 1]
        root = tree[index]

        hpj = J[layer, root] + h[:, layer]
        hmj = -J[layer, root] + h[:, layer]

        fe_tree += -torch.log(2 * (torch.cosh(beta*hpj) * torch.cosh(beta*hmj)).sqrt()).sum(dim=1) / beta
        for k in range(len(root)):
            h[:, root[k]] += torch.log(torch.cosh(beta*hpj) / torch.cosh(beta*hmj))[:, k] / (2*beta)
        tree = tree[len(layer):]

    return fe_tree
