# import torch
from __future__ import print_function
import numpy as np
import torch

a = np.ones(5)
b = torch.from_numpy(a)

print(a,b)

np.add(a, 1, out=a)
print(a)

# Uninitialized matrix is declared (whatever values appear in memory allocated will initialize the matrx)
x = torch.empty(5,3)
print('Uninitialized matrix:\n', x)

# Construct a random matrix
x = torch.rand(5,3)
print('torch.rand() :\n', x)

# Construct a matrix filled zeros and of dtype long:
x = torch.zeros(5,3, dtype=torch.long)
print('torch.zeros():\n', x)

# Construct a matrix from data
x = torch.tensor([5.5, 3])
print('tensor matrix from data:\n', x)
arr = np.arange(2, 5, 0.2)
x = torch.tensor(arr)
print('tensor matrix from data arr:\n', x)

# Create a tensor based on an existing tensor. These methods will reuse the properties
#    of the input tensor, e.g. dtype, unless new values are provided by user
x = torch.rand(5,3)
x = x.new_ones(5,3, dtype=torch.double)
print("x.new_ones():\n", x)

x = torch.rand_like(x, dtype=torch.float)
print('torch.rand_like():\n', x)

# Get the size of tensor
print('size of tensor, x.size():\n', x.size(), 'dtype: ', type(x.size()) )
# actually is tuple so support all tuple operations

# ===========================
# check if is tensor
print(torch.is_tensor(x))
# check if is floating_point
print(torch.is_floating_point(x))

# set and get torch-tensor default type
# torch.set_default_dtype(d)
# torch.get_default_dtype() → torch.dtype

print("total number of elements:\n", torch.numel(x))

# *** torch.set_flush_denormal(mode) → bool : Disables denormal(异常的) floating numbers on CPU.

# create new torch tensors
# Random sampling creation ops are listed under Random sampling and include: torch.rand()
#   torch.rand_like() torch.randn() torch.randn_like() torch.randint() torch.randint_like()
#   torch.randperm() You may also use torch.empty() with the In-place random sampling methods
#   to create torch.Tensor s with values sampled from a broader range of distributions.
# such as
x.new_empty(x.size())
print(x)

# Warning
# Random sampling creation ops are listed under Random sampling and include: torch.rand()
#   torch.rand_like() torch.randn() torch.randn_like() torch.randint() torch.randint_like()
#   torch.randperm() You may also use torch.empty() with the In-place random sampling methods
#   to create torch.Tensor s with values sampled from a broader range of distributions.
# such as,
x = torch.ones(2,3)
y = x.clone()
y = y+1
print('*'*20, '\n', x, '\n', y)


y = x.detach()
y = y+1
print(x, '\n', y)

y = torch.tensor(x) # this trigers a warning
y += 1
print(x, '\n', y)

# \\\\\\\\ Numpy Bridge \\\\\\\\\
# Converting a Torch Tensor to a NumPy Array
a =torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a,b) # changing the np array changed the Torch Tensor automatically

# All the Tensors on the CPU except a CharTensor support converting to NumPy and back.

##### CUDA Tensors
# Tensors can be moved onto any device using the .to method.
if torch.cuda.is_available():
    device = torch.device("cuda")  # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)  # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))  # ``.to`` can also change dtype together!



