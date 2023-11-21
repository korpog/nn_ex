import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

shape = (2, 2,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)

# print(f"shape: {ones_tensor.shape} \n" +
#       f"dtype: {ones_tensor.dtype} \n device: {ones_tensor.device}")

# print(rand_tensor)
# print(rand_tensor[0]) # first row
# print(rand_tensor[:, 1]) # second column

# print(rand_tensor)
# print(ones_tensor)
# t = torch.cat([rand_tensor, ones_tensor], dim=1)
# print(t)

# matrix multiplication
T = rand_tensor @ ones_tensor
T1 = rand_tensor @ rand_tensor.T
print(T1)
T1.add_(3)
print(T1)
