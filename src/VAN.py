import torch
import time
import numpy as np
from torch import nn, optim
from utils import energy_ising, sum_up_tree


class MaskedLinear(nn.Linear):
    """ masked linear layer """

    def __init__(self, in_channels, out_channels, n, bias, selfconnection, mask):
        super(MaskedLinear, self).__init__(in_channels * n, out_channels * n, bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n
        self.selfconnection = selfconnection

        self.register_buffer('mask', mask)
        # exclusive determines whether this masked linear layer has auto-connections or not
        if self.selfconnection:
            self.mask = torch.tril(self.mask) - torch.eye(n)
        else:
            self.mask = torch.tril(self.mask)
        self.mask = torch.cat([self.mask] * in_channels, dim=1)
        self.mask = torch.cat([self.mask] * out_channels, dim=0)
        self.weight.data *= self.mask

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)

    def extra_repr(self):
        return (super(MaskedLinear, self).extra_repr() +
                ', exclusive={exclusive}'.format(**self.__dict__))


class MADE(nn.Module):
    def __init__(self, D, net_depth, net_width, mask):
        super().__init__()
        self.D = D
        self.net_depth = net_depth
        self.net_width = net_width

        # set quantities to enforce root vetices free
        self.register_buffer('free_top0', torch.ones(self.D))
        self.register_buffer('free_top1', torch.zeros(self.D))
        self.free_top0[0] = 0
        self.free_top1[0] = 0.5

        # calculate connections in this network
        self.connections = len(torch.tril(mask).nonzero())

        # define a simple MLP neural net
        self.net = []
        hs = [1] + [self.net_width]*self.net_depth + [1]
        for l in range(len(hs)-1):
            self.net.extend(
                [MaskedLinear(hs[l], hs[l+1], D, 1, 0 if l == 0 else 1, mask),
                 nn.Tanh()])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net.extend([nn.Sigmoid()])  # add the sigmoid function to the output layer
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        xhat = self.net(x)
        xhat = xhat * self.free_top0 + self.free_top1
        return xhat

    def sample(self, input, i):
        """
        fast sampling of ouput configurations at vertices i without going through
        all the network
        """

        index = i
        L = self.net_depth
        inter = self.net[:2 * L](input)
        weight = (self.net[2 * L].weight * self.net[2 * L].mask)[index]
        output = inter.matmul(weight.t()) + self.net[2*L].bias[index]
        output = torch.sigmoid(output)
        return output


def VAN(args):
    if args.dtype == 'float32':
        dtype = torch.float32
    elif args.dtype == 'float64':
        dtype = torch.float64
    else:
        raise Exception('Unknown dtype.')

    J = np.loadtxt('../data/energy_function{}nodes.txt'.format(args.D))
    J = torch.from_numpy(J).to(dtype).to(args.device)

    if args.method == 'FVS':
        # read the FVS nodes and tree hierarchy
        FVS = np.loadtxt('../data/' + 'fvs{}nodes.txt'.format(args.D)).astype(np.int)
        with open('../data/' + 'trees{}nodes.txt'.format(args.D)) as f:
            list1 = f.readlines()
        f.close()
        tree = []
        tree_hierarchy = []
        for i in range(len(list1)):
            current_line = list(map(int, list1[i].split()))
            tree += current_line
            tree_hierarchy.append(current_line)
        n = len(FVS)
        mask = torch.from_numpy(
            np.loadtxt('../data/' + args.method + 'adj{}nodes.txt'.format(args.D))
        ).float()
    elif args.method == 'chordal':
        n = args.D
        mask = torch.from_numpy(
            np.loadtxt('../data/' + args.method + 'adj{}nodes.txt'.format(args.D))
        ).float()
    elif args.method == 'dense':
        n = args.D
        mask = (torch.ones([n] * 2) - torch.eye(n)).float()
    else:
        raise Exception('Unknown method.')

    model = MADE(n, args.net_depth, args.net_width, mask).to(dtype).to(args.device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    start_time = time.time()

    # run epoch
    torch.set_grad_enabled(True)
    beta = args.beta
    while beta <= args.beta_to:
        for epoch in range(args.num_epochs):
            sample_size = args.sample

            # sampling
            sample = torch.zeros([sample_size, n], device=args.device, dtype=dtype)
            with torch.no_grad():
                for i in range(n):
                    if not isinstance(i, list):
                        i = [i]
                    xhat = model.sample(sample, i)
                    sample[:, i] = torch.bernoulli(xhat) * 2 - 1

            xhat = model(sample)

            # calculate effective energy of tree nodes
            if args.method == 'FVS':
                energy_tree = sum_up_tree(sample, J, FVS, tree, tree_hierarchy,
                                          sample_size, beta, args.device)

            # calculate entropy, energy, free energy and loss
            entropy = - (torch.log(xhat + 1e-10) * (sample + 1) + torch.log(1 - xhat + 1e-10) * (1 - sample)) / 2
            entropy = torch.sum(entropy, dim=1)

            with torch.no_grad():
                energy0 = energy_ising(sample, J[FVS][:, FVS], n) + energy_tree \
                    if args.method == 'FVS' else energy_ising(sample, J, n)
                free_energy = - entropy / beta + energy0

            loss = torch.mean(- entropy * (free_energy - free_energy.mean()))

            # backprop ang update
            opt.zero_grad()
            loss.backward()
            opt.step()
        beta += args.beta_inc

    # calculate quantities use bigger sample size(calc_number)
    torch.set_grad_enabled(False)
    sample_size = args.calc

    # sampling
    sample = torch.zeros([sample_size, n], device=args.device, dtype=dtype)
    with torch.no_grad():
        for i in range(n):
            if not isinstance(i, list):
                i = [i]
            xhat = model.sample(sample, i)
            sample[:, i] = torch.bernoulli(xhat) * 2 - 1

    xhat = model(sample)

    if args.method == 'FVS':
        energy_tree = sum_up_tree(sample, J, FVS, tree, tree_hierarchy,
                                  sample_size, args.beta_to, args.device)

    energy0 = energy_ising(sample, J[FVS][:, FVS], n) + energy_tree \
        if args.method == 'FVS' else energy_ising(sample, J, n)
    entropy = - (torch.log(xhat + 1e-10) * (sample + 1) + torch.log(1 - xhat + 1e-10) * (1 - sample)) / 2
    entropy = torch.sum(entropy, dim=1)
    free_energy = - entropy / args.beta_to + energy0

    times = time.time() - start_time

    # count config
    sample_list = sample.tolist()
    energy_list = energy0.tolist()
    config = []
    nums = []
    energy = []
    for i in range(len(sample_list)):
        if sample_list[i] not in config:
            config.append(sample_list[i])
            energy.append(energy_list[i])
            nums.append(1)
        else:
            index = config.index(sample_list[i])
            nums[index] += 1

    return -free_energy.mean().cpu().numpy()/args.D, entropy.mean().cpu().numpy(), times, config, energy, nums

