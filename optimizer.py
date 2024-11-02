from torch import nn
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import MaxPool2d, Flatten, Linear, Sequential
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1, drop_last=True)


# 使用sequential
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(), lr=0.01, )
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss  # running_loss相当于扫一遍全部数据的loss总和
    print(running_loss)
# for data in dataloader循环：
# 这个循环通常用于遍历数据集中的每个批次（batch）数据。
# dataloader是一个用于批次化处理数据的工具，它会将数据集按照指定的批次大小分割，并提供数据加载的迭代器。
# 在每次迭代中，for data in dataloader会从数据加载器中获取一个批次的数据，然后你可以对这个批次的数据进行前向传播、计算损失、反向传播和参数更新等操作。
# 这个循环通常嵌套在训练循环中，用于处理每个训练批次的数据。
#
# for epoch in range(X)循环：
# 这个循环用于控制整个训练过程的迭代次数，其中X代表训练的总轮数（epochs）。
# 一个epoch表示将数据集中的所有样本都用于训练一次，通常情况下，训练过程会重复多个epoch以便模型能够更好地学习数据的特征。
# 在每个epoch循环中，你会执行多次for
# data in dataloader循环，每次处理一个批次的数据，并进行前向传播、损失计算、反向传播和参数更新等训练步骤。
# 一般来说，训练过程会在每个epoch结束时进行模型评估，例如计算验证集上的准确率或损失，以便监控模型的训练情况和避免过拟合。


