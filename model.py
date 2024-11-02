from torch import nn
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import MaxPool2d, Flatten, Linear
import torch

#搭建神经网络
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


