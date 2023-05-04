import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def make_features(x):
    '''
    Build features i.e. a matrix with columns [x,x^2,x^3]
    :param x:
    :return:
    '''
    # (32,) -> (32,1)
    x = x.unsqueeze(1)
    # print("x:", x)
    features = torch.cat([x ** i for i in range(1, 4)], 1)
    # print("features:", features)
    return features


# [0.5, 3, 2.4] -> [[0.5000],[3.0000],[2.4000]]
# shape: (3,) -> (3,1)
W_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
# print(W_target)
b_target = torch.FloatTensor([0.9])


# print(b_target)

def f(x):
    # [32, 3] * [3,1] + [32,1] -> [32,1]
    return x.mm(W_target) + b_target[0]


def get_batch(batch_size=32):
    random = torch.arange(0.1, 3.3, 0.1, dtype=torch.float32)
    # random = torch.abs(torch.randn(batch_size))
    print('random:', random)
    x = make_features(random)
    y = f(x)
    if torch.cuda.is_available():
        return torch.Tensor(x).cuda(), torch.Tensor(y).cuda()
    else:
        return torch.Tensor(x), torch.Tensor(y)


x, y = get_batch()


# print(x)
# print(x.shape)  # torch.Size([32, 3])
# print(y)
# print(y.shape)  # torch.Size([32, 1])


# 定义多项式回归模型
class PolyModel(nn.Module):
    def __init__(self):
        super(PolyModel, self).__init__()
        self.poly = nn.Linear(3, 1)

    def forward(self, x):
        out = self.poly(x)
        return out


# 如果GPU可用，将模型转移到GPU上
if torch.cuda.is_available():
    model = PolyModel().cuda()
else:
    model = PolyModel()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# 训练模型
num_epochs = 10000
for epoch in range(num_epochs):
    # 如果GPU可用，将数据转移到GPU上
    if torch.cuda.is_available():
        inputs = torch.Tensor(x).cuda()
        targets = torch.Tensor(y).cuda()
    else:
        inputs = torch.Tensor(x)
        targets = torch.Tensor(y)

    # forward
    out = model(inputs)
    loss = criterion(out, targets)
    # backward
    optimizer.zero_grad()
    loss.backward()
    # 更新模型的参数
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}],Loss:{loss.data.item()}')
        for name, param in model.named_parameters():
            # 打印参数值
            print(f"{name}: {param.data}")

# 将模型和数据再移动到CPU中
model = model.to("cpu")
x_train = x.to("cpu")
y_train = y.to("cpu")
print(x_train)
print(y_train)
model.eval()
predict = model(torch.Tensor(x_train))
predict = predict.data.numpy()
print(predict)
fig, ax = plt.subplots()
ax.plot(torch.arange(0.1, 3.3, 0.1, dtype=torch.float32).numpy(), y_train.numpy(), 'ro', label='Original Data')
ax.plot(torch.arange(0.1, 3.3, 0.1, dtype=torch.float32).numpy(), predict, 'b', label='Fitting Line')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('x vs y plot')

# ax.set_xlim([-1.5, 1.5])    # 设置 x 轴范围
plt.show()

# plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original Data')
# plt.plot(x_train.numpy(), predict, 'b', label='Fitting Line')
# plt.show()

#
# epoch = 0
# while True:
#     batch_x, batch_y = get_batch()
#     output = model(batch_x)
#     loss = criterion(output, batch_y)
#     print_loss = loss.data.item()
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     epoch += 1
#     if print_loss < 1e-3:
#         break
#
# print(f'Loss:{loss},after {epoch} epochs')
# for name, param in model.named_parameters():
#     print(f"{name}: {param.data}")
