import torch
from torch.autograd import Variable

# Pytorch的变量可以支持自动求导，本质上和Tensor没有区别，不过前者会被放入一个计算图中，然后
# 可以进行前向传播、反向传播、自动求导

a = torch.Tensor([[2, 4], [4, 9], [7, 6]])
v_a = torch.autograd.Variable(a)
print(v_a)
print(type(v_a))

# Create Variable
# x = torch.tensor([1.], requires_grad=True)
x = Variable(torch.Tensor([1]), requires_grad=True)
w = torch.tensor([2.], requires_grad=True)
b = torch.tensor([3.], requires_grad=True)

# Build a computational graph
y = w * x + b

# Compute gradients
y.backward()

# Print out the gradients
print(x.grad)  # tensor([2.])
print(w.grad)  # tensor([1.])
print(b.grad)  # tensor([1.])

# 上面是对标量进行求导，下面是对矩阵进行求导
x = torch.tensor([1., 2, 3], requires_grad=True)
# x = Variable(x, requires_grad=True)
print(x)

y = x * x
# print(y)

gradient=torch.tensor([1.0,1.0,1.0])
# y.backward()   # 这样写就会报错
# y.backward(torch.FloatTensor([1,0.1,0.01]))  # 原本的梯度再去×[1,0.1,0.01]
y.backward(gradient)

print(x.grad)