import torch
def nmfiter(A,marg):
    return torch.softmax(A @ marg)
