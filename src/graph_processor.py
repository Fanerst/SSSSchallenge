import numpy as np
import networkx as nx


def save_graph(G):
    n = G.number_of_nodes()
    edge = list(G.edges)

    with open('../data/{}nodes.txt'.format(n), 'w') as f:
        f.write('{} {}\n'.format(n, len(edge)))
        for line in edge:
            f.write('{} {}\n'.format(line[0], line[1]))
    f.close()

    J = np.zeros([n, n])
    for e in edge:
        J[e[0], e[1]] = J[e[1], e[0]] = -1
    np.savetxt('../data/energy_function{}nodes.txt'.format(n), J, fmt='%f')


def chordal_completion(G, method, order=None):
    """
    there are two choices of method: FVS and chordal
    FVS stands for chordal completing FVS nodes
    chordal stands for chordal completing the whole graph
    """
    n = G.number_of_nodes()
    G1 = G.copy()
    if not order:
        order = np.arange(G1.number_of_nodes()).tolist()
    nodes = order[::-1]
    for i in range(G1.number_of_nodes()):
        subnodes = list(set(nodes[i:]).intersection(set(G1.neighbors(nodes[i]))))
        for j in range(len(subnodes)):
            for k in range(j+1, len(subnodes)):
                if not subnodes[k] in G1.neighbors(subnodes[j]):
                    G1.add_edge(subnodes[k], subnodes[j])

    ADJ = nx.adjacency_matrix(G1, nodelist=order).todense()
    np.savetxt('../data/'+ method + 'adj{}nodes.txt'.format(n), ADJ, fmt='%f')


def FVS_decomposition(G, rng):
    n = G.number_of_nodes()
    G1 = G.copy()
    fvs = []
    while G1.number_of_nodes():
        flag = True
        while flag:
            temp = []
            flag = False
            for i in list(G1.node):
                if G1.degree[i] <= 1:
                    temp.append(i)
                    flag = True
            if not flag:
                break
            G1.remove_nodes_from(temp)
        if not G1.number_of_nodes():
            break
        degrees = np.array(G1.degree)
        degree_max = degrees[rng.choice(np.where(degrees[:, 1] == max(degrees[:, 1]))[0]), 0]
        fvs.append(degree_max)
        G1.remove_node(degree_max)

    G1 = G.copy()
    G1.remove_nodes_from(fvs)
    ccs = list(nx.connected_components(G1))
    trees = {}.fromkeys(np.arange(len(ccs)))
    for key in trees.keys():
        trees[key] = []
    for l in range(len(ccs)):
        tree = G.subgraph(ccs[l]).copy()
        while tree.number_of_nodes():
            temp = []
            for i in list(tree.node):
                if tree.number_of_nodes() == 1 or tree.number_of_nodes() == 2:
                    node = i
                    temp.append(i)
                    break
                if tree.degree[i] == 1:
                    temp.append(i)
            tree.remove_nodes_from(temp)
            trees[l].append(temp)

    tree = []
    max_length = 0
    for key in trees.keys():
        l = len(trees[key])
        if l >= max_length:
            max_length = l

    for i in range(max_length):
        tree.append([])
        for key in trees.keys():
            if i < len(trees[key]):
                tree[i] += trees[key][i]

    G_fvs = nx.Graph()
    G_fvs.add_nodes_from(fvs)
    for i in range(len(fvs)):
        for j in range(i+1, len(fvs)):
            if nx.has_path(G, fvs[i], fvs[j]):
                G_fvs.add_edge(fvs[i], fvs[j])

    chordal_completion(G_fvs, 'FVS', fvs)

    with open('../data/fvs{}nodes.txt'.format(n), 'w') as f:
        for i in fvs:
            f.write(str(i) + ' ')
    f.close()

    with open('../data/trees{}nodes.txt'.format(n), 'w') as f:
        for l in range(len(tree)):
            for i in tree[l]:
                f.write(str(i) + ' ')
            f.write('\n')
    f.close()


if __name__ == '__main__':
    adj = np.loadtxt('../data/adj60nodes.txt')
    G = nx.from_numpy_matrix(adj)
    rng = np.random.RandomState(78825)
    save_graph(G)
    chordal_completion(G, 'chordal')
    FVS_decomposition(G, rng)
