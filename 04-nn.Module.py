# import torch
# from torch import nn
# # 所有的层和损失函数都是在 torch.nn，所有的模型构架都是从基类 nn.Module 继承的
# class my_net(nn.Module):
#     def __init__(self,other_args):
#         super(my_net, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size)
#         # othre layers
#
#     def forward(self,x):
#         x = self.conv1(x)
#         return x
#
# criterion = nn.CrossEntropyLoss()
# loss = criterion(output,target)
#
# 优化器
# 
# import torch.optim
# optimizer.zeros()
# optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
# loss.backward()
# optimizer.step()
