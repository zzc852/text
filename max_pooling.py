import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import MaxPool2d
#最大池化层
dataset = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


# 搭建网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


writer = SummaryWriter("logs")  # 日志文件存储位置
tudui = Tudui()
print(tudui)

step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print("原图像的形状", imgs.shape)
    print("池化之后图像的形状", output.shape)
    writer.add_images("maxpool_input", imgs, step)
    writer.add_images("maxpool_output", output, step)
    step = step + 1
writer.close()
