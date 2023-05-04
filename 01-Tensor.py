import torch

a = torch.Tensor([[2,4],[4,9],[7,6]])
print(a)
print(a.dtype)  # torch.float32
print(a.shape) # torch.Size([3, 2])


b = torch.zeros(3,2)
print(b)

# 取正态分布作为随机值
c = torch.randn(2,4)
print(c)

a[0,1] = 100  # 修改tensor中指定位置的值
print(a)

# Tensor与Numpy的转化
n_a = a.numpy()
print(n_a)
print(type(n_a))  # <class 'numpy.ndarray'>

t_a = torch.from_numpy(n_a)
print(t_a)
print(type(t_a)) # <class 'torch.Tensor'>

# 判断是否可以支持GPU
print(torch.cuda.is_available())  # True