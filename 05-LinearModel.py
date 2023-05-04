import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable

# 随机生成一组特征值
feature = np.random.rand(100)

# 设定一个线性关系，并生成带有噪声的目标值
target = 2 * feature + 3 + np.random.randn(100) * 0.2

print("Target:  ", target)
# 将numpy数组改为二维的
x_train = feature.reshape(-1, 1)
y_train = target.reshape(-1, 1)

# 绘制散点图
plt.scatter(feature, target)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

# 打印前10个样本
print("x_train: {}".format(x_train[:10]))
print("y_train: {}".format(y_train[:10]))

# 将numpy数组数据类型转为float32（GPU处理）再转为tensor
x_train = torch.from_numpy(x_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))


# 建立线性模型
class MyLinearModel(nn.Module):
    def __init__(self):
        super(MyLinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入和输出都是一维的

    def forward(self, x):
        out = self.linear(x)
        return out


# 如果GPU可用，将模型转移到GPU上
if torch.cuda.is_available():
    model = MyLinearModel().cuda()
else:
    model = MyLinearModel()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# 训练模型
num_epochs = 100000
for epoch in range(num_epochs):
    # 如果GPU可用，将数据转移到GPU上
    if torch.cuda.is_available():
        inputs = Variable(x_train).cuda()
        targets = Variable(y_train).cuda()
    else:
        inputs = Variable(x_train)
        targets = Variable(y_train)

    # forward
    out = model(inputs)
    loss = criterion(out, targets)
    # backward
    optimizer.zero_grad()
    loss.backward()
    # 更新模型的参数
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}],Loss:{loss.data.item()}')
        for name, param in model.named_parameters():
            # 打印参数值
            print(f"{name}: {param.data}")

# 将模型和数据再移动到CPU中
model = model.to("cpu")
x_train = x_train.to("cpu")
y_train = y_train.to("cpu")


model.eval()
predict = model(Variable(x_train))
predict = predict.data.numpy()
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original Data')
plt.plot(x_train.numpy(), predict, 'b', label='Fitting Line')
plt.show()

# 使用单个值进行预测
# predict = model(torch.Tensor([[3]]))
# predict = predict.data.numpy()
# print(predict)  # [[9.063663]]

