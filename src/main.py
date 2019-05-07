from VAN import VAN
from exact import exact_logZ
from utils import get_args


def answer1():
    device = 'cuda:0'
    result_exact, time_exact = exact_logZ(60, 1, device)
    args = get_args('FVS', 1, 1, 1, device)
    result_FVS, times_FVS, _, _, _ = VAN(args)
    args = get_args('chordal', 0, 1, 1, device)
    result_chordal, times_chordal, _, _, _ = VAN(args)
    args = get_args('dense', 0, 1, 1, device)
    result_dense, times_dense, _, _, _ = VAN(args)

    print(result_exact, time_exact)
    print(result_FVS, times_FVS)
    print(result_chordal, times_chordal)
    print(result_dense, times_dense)


def answer2():
    beta = 100
    device = 'cuda:0'
    args = get_args('chordal', 0, 1, beta, device)
    _, _, config, energy, nums = VAN(args)
    print(energy)
    print(nums)

if __name__ == '__main__':
    answer1()
    answer2()


