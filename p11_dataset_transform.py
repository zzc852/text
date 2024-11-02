import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
#创建tensorBoard写入器
writer = SummaryWriter("p10") # 日志文件存储位置，TensorBoard 会在该目录下创建一个日志文件以记录训练数据。
#transforms.Compose 是用于将多个数据转换组合在一起的工具。
#transforms.ToTensor() 将输入图像转换为 PyTorch 张量（Tensor），并将图像像素值归一化到 [0, 1] 的范围。
dataset_transform = transforms.Compose([
    transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="./dataset1", train=True,transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset1", train=False,transform=dataset_transform, download=True)

print(test_set[0])
img, target = test_set[0]  # target对应类的编号 对应cat
print(img)
print(target)
print(test_set.classes[target])

for i in range(10):
    img, target = test_set[i]
    writer.add_image("torchvision",img,i)#将图片添加到tensorBoard日志中





