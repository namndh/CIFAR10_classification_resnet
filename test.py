import torch 

x = torch.Tensor(5,3)
print(x)

y = torch.rand(5,3)
print(y)

print("Size of x:{}, y:{}".format(x.size(), y.size()))
print(x + y)