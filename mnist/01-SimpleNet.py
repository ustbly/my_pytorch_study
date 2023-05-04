import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import simpleNet

if __name__ == '__main__':
    # Hyperparameters
    batch_size = 64
    learning_rate = 1e-2
    num_epochs = 20

    # 数据预处理的组合操作
    data_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )

    # 下载训练数据集
    train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)

    # 解决pytorch训练的时候GPU利用率不能跑满的情况：
    '''
        1. DataLoader添加num_workers参数增加多个进程（此参数在Windows上打开必须将代码放到main函数才能正常执行）
        2. DataLoader添加pin_memory参数，省掉了将数据从CPU传入到缓存RAM里面，再给传输到GPU上；为True时是直接映射到GPU的相关内存块上

    '''
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = simpleNet(28 * 28, 300, 100, 10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # 训练网络
    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            # 将数据也移动到GPU上
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = inputs.view(inputs.size(0), -1)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(inputs), len(train_loader.dataset),
                           100. * i / len(train_loader), loss.item()))

        avg_loss = total_loss / len(train_loader)
        print('Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))

    # 测试模型
    model.eval()
    eval_acc = 0.0
    eval_loss = 0.0
    for inputs, labels in test_loader:
        # 将数据也移动到GPU上
        inputs = inputs.to(device)
        labels = labels.to(device)

        inputs = inputs.view(inputs.size(0), -1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.step()

        eval_loss += loss.item() * labels.size(0)
        _, pred = torch.max(outputs, 1)
        num_correct = (pred == labels).sum()
        eval_acc += num_correct.item()
    print(f"loss: {eval_loss / len(test_dataset)},acc: {eval_acc / len(test_dataset)}")



