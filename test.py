import torch
import numpy as np


b = np.array([[[1,1,1],[2,2,2]],[[1,1,1],[4,4,1]],[[1,1,1],[1,1,1]]])
print(a.shape)
print(b.shape)
a = np.array([[1,2],[1,3],[1,4]])
a = torch.from_numpy(a)
b = torch.from_numpy(b)

print(a.shape)
print(b.shape)

print((a.unsqueeze(2) * b).sum(dim=1))