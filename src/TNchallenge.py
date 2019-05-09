import torch
import numpy as np
import time


start = time.time()
bM = torch.DoubleTensor([[np.exp(-1), np.exp(1)],[np.exp(1), np.exp(-1)]])
U, S, V = torch.svd(bM)
sq_S = torch.sqrt(torch.diag(S))
B1 = U @ sq_S
B2 = sq_S @ V.t()                                                                                                        #1])  # A3=torch.einsum("ij,ab,cd,jbd->iac",[B,B,B,I3])
I3 = torch.zeros(2, 2, 2).double()
I3[0][0][0] = I3[1][1][1] = 1
point_3 = torch.einsum("ij,kl,djk->dil",[B2,B1,I3])
point_5 = torch.einsum("aij,bjk,ckl,dlm,emi->abcde",[point_3 for i in range(5)])
l1 = [B2 for i in range(5)]
l1.append(point_5)
# 22222
top = torch.einsum("ia,jb,kc,ld,me,abcde->ijklm",l1)
# 12111
l2 = [B1, B2, B1, B1, B1]
l2.append(point_5)
mid_up1 = torch.einsum("ai,jb,ck,dl,em,abcde->ijklm",l2)
l3 = [mid_up1 for i in range(5)]
mid_up2 = torch.einsum("abklc,dcmne,feopg,hgqri,jistb->adfhjklmnopqrst",l3)
up = torch.einsum("abcde...,abcde->...",[mid_up2,top])
#12222
l4 = [B1, B2, B2, B2, B1]
l4.append(point_5)
mid_down1 = torch.einsum("ai,jb,kc,ld,em,abcde->ijklm",l4)
l5 = [mid_down1 for i in range(5)]
mid_down2 = torch.einsum("abklc,dcmne,feopg,hgqri,jistb->adfhjklmnopqrst",l5)
down = torch.einsum("abcde...,abcde->...",[mid_down2,top])
all = torch.einsum("abcdefghij,bcdefghija",[up,down])
result = torch.log(all)/60

times = time.time()-start
result = result.item()

print(result, times)
