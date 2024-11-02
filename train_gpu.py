import torch
from torch.nn import Flatten, MaxPool2d, Conv2d
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
# 准备数据集
train_data = torchvision.datasets.CIFAR10("dataset1",train=True, transform=torchvision.transforms.ToTensor(),
                                         download=True)
test_data = torchvision.datasets.CIFAR10("dataset1",train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
# len()获取数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=64, drop_last=True)

#创建网络模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            MaxPool2d(2),
            Conv2d(32,32,5,1,2  ),
            MaxPool2d(2),
            Conv2d(32,64,5,1,2),
            MaxPool2d(2),
            Flatten(),#平摊到一条线上
            nn.Linear(64*4*4,64),
            nn.Linear(64,10),

    )


    def forward(self, x):
       x=self.model(x)
       return x

tudui = Tudui()
if torch.cuda.is_available():
   tudui=tudui.cuda() #利用gpu

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
   loss_fn=loss_fn.cuda() # 利用gpu

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0

# 记录测试的次数
total_test_step = 0

# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print("-------------第 {} 轮训练开始------------".format(i+1))
    #训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
         imgs=imgs.cuda() #利用gpu
         targets=targets.cuda()#利用gpu
        output = tudui(imgs)
        loss = loss_fn(output, targets)

        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 ==0:
            print("训练次数：{}， Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss",loss.item(), total_train_step)

    # 测试步骤开始
    total_test_loss = 0
    total_accuracy=0    #整体的正确率个数
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
             imgs = imgs.cuda()  # 利用gpu
             targets = targets.cuda()  # 利用gpu
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy=(outputs.argmax(1) ==targets).sum() #求出每个位置最大的值==与正确的比较  之后求和
            total_accuracy=total_accuracy+accuracy  #正确率的个数加一起
    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss",total_test_loss, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")
writer.close()