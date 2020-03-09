# import torch
from __future__ import print_function
import numpy as np
import torch

'''
Torch tensor operations
'''

# Addition
# syntax 1
x = torch.rand(5,3)
y = torch.rand(5,3)
print(x+y)

# syntax 2
print(torch.add(x, y))
#   it provides an output tensor as argument
result = torch.empty(x.size())
torch.add(x,y,out=result)
print(result)

# addition: in-place
y.add_(x)
print(y)

#  any operation that mutates a tensor in-place is post-fixed with an
#    '_', e.x.: x.copy_(y), x.t_(), will CHANGE x
z = torch.empty(y.size())
z.copy_(y)
z = z + 1
print(y,z)
print(z.t_()) # matrix transpose

# Resizing: if you want to resize/reshape tensor, you can use torch.view:
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1,8) # the size -1 is inferred from other dimensions, here is 8 (16/8, so it's 2)
print(x.size(), y.size(),z.size())

# ** If you have a one element tensor, use .item() to get the value as a Python number**
x = torch.randn(1)
print(x)
print(x.item())
# to get all of an array, use
# x = torch.randn(2,3)
# y = np.zeros(x.size())
#
# y = [e.item() for e in x]
# print(y)
