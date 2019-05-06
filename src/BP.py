import numpy as np
from utils import readgraph
import time


def BP_revised(D, beta):
    start_time = time.time()
    num_edges, edges, neighbors, J = readgraph(D)

    stepmax = 10000
    epsilon = 1e-4
    difference_max = 10
    damping_factor = 0

    h = np.random.randn(D, D)
    # belief propagation
    for step in range(stepmax):
        for i in range(D):
            for j in range(len(neighbors[i])):
                a = neighbors[i][j]
                B = list(neighbors[i])
                B.remove(a)
                temp = (np.arctanh(
                        np.tanh(beta * J[i, B]) * np.tanh(beta * h[B, i])
                        ) / beta).sum()
                temp = damping_factor*h[i][a] + (1-damping_factor)*temp
                difference = abs(temp - h[i][a])
                h[i][a] = temp
                if i == 0 and j == 0:
                    difference_max = difference
                elif difference > difference_max:
                    difference_max = difference
        if difference_max <= epsilon:
            break

    # calculate free energy
    fe_node = np.zeros(D)
    for i in range(D):
        B = list(neighbors[i])
        temp1 = (np.cosh(beta * (J[i, B] + h[B, i])) /
                np.cosh(beta * h[B, i])).prod()
        temp2 = (np.cosh(beta * (-J[i, B] + h[B, i])) /
                np.cosh(beta * h[B, i])).prod()
        fe_node[i] = - np.log(temp1 + temp2) / beta
    fe_node_sum = np.sum(fe_node)

    fe_edge = np.zeros(num_edges)
    edge_count = 0
    for edge in edges:
        i, j = edge
        temp1 = np.exp(beta*J[i,j]) * np.cosh(beta*(h[i,j]+h[j,i])) + \
                np.exp(-beta*J[i,j]) * np.cosh(beta*(h[i,j]-h[j,i]))
        temp2 = 2*np.cosh(beta*h[i,j])*np.cosh(beta*h[j,i])
        fe_edge[edge_count] = - np.log(temp1/temp2) / beta
        edge_count += 1
    fe_edge_sum = np.sum(fe_edge)

    fe_sum = fe_node_sum - fe_edge_sum

    # calculate energy
    energy_BP = np.zeros(num_edges)
    edge_count = 0
    for edge in edges:
        i, j = edge
        temp1 = -J[i,j]*np.exp(beta*J[i,j])*np.cosh(beta*(h[i,j]+h[j,i])) + \
                J[i,j]*np.exp(-beta*J[i,j])*np.cosh(beta*(h[i,j]-h[j,i]))
        temp2 = np.exp(beta*J[i,j])*np.cosh(beta*(h[i,j]+h[j,i])) + \
                np.exp(-beta*J[i,j])*np.cosh(beta*(h[i,j]-h[j,i]))
        energy_BP[edge_count] = temp1 / temp2
        edge_count += 1
    energy_BP = np.sum(energy_BP)

    # calculate entropy
    entropy_BP = beta*(energy_BP - fe_sum)

    # calcualte magnetzation
    mag_BP = np.zeros(D)
    for i in range(D):
        B = list(neighbors[i])
        temp = np.arctanh(
                np.tanh(beta*J[i, B]) * np.tanh(beta*h[B,i])
                ).sum()
        mag_BP[i] = np.tanh(temp)

    # calculate connected correlation
    correlation_BP = np.empty(num_edges)
    edge_count = 0
    for edge in edges:
        i, j = edge
        temp1 = np.exp(beta*J[i,j])*np.cosh(beta*(h[i,j]+h[j,i]))
        temp2 = np.exp(-beta*J[i,j])*np.cosh(beta*(h[i,j]-h[j,i]))
        correlation_BP[edge_count] = (temp1 - temp2) / (temp1 + temp2) - \
        mag_BP[i] * mag_BP[j]
        edge_count += 1

    times = time.time() - start_time

    return -fe_sum/D, energy_BP, entropy_BP, mag_BP, correlation_BP, step, times


if __name__ == '__main__':
    logZ, e, entropy, mag, cor, step, times = BP_revised(60, 1)
    print(step)
    print(times)
    print(logZ, e, entropy)

