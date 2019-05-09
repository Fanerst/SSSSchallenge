from itertools import combinations
import numpy as np

with open('../data/12nodes.txt') as f:
    list1 = f.readlines()

edges = []
for i in range(len(list1)):
    if i > 0:
        a, b = list1[i].split()
        edges.append([int(a), int(b)])

combines = list(combinations(edges, 6))

list1 = []
nums = 0
for i in range(len(combines)):
    for j in range(len(combines[i])):
        list1 += combines[i][j]
    list1.sort()
    if list1 == np.arange(12).tolist():
        nums += 1
    list1 = []

print(nums*2*2**6)