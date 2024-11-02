import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset1", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


# 搭建网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x





writer = SummaryWriter("p9")  # 日志文件存储位置
tudui = Tudui()
print(tudui)

step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print("原图像的形状", imgs.shape)
    print("卷积之后图像的形状", output.shape)
    writer.add_images("input", imgs, step)
    # 卷积之后图像的形状 torch.Size([64, 6, 30, 30])是6个通道的 而add_images只能接收3通道的输入
    output = torch.reshape(output, (-1, 3, 30, 30))  # 不严谨操作 ---对output进行reshape 增大batchsize的数量 减少通道数
    writer.add_images("Conv_output", output, step)
    step = step + 1
writer.close()
