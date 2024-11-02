from torch.utils.data import Dataset
from PIL import Image
import os
#加载数据集
#dataset 提供一种方式获取数据及其label模型最终要输出的东西，回归里是一个个类别  预测里是一个个数值
#如何获取每一个数据及其label
#告诉我们总共有多少的数据
#dataloader   为网络提供不同的数据形式

class MyData(Dataset):
      def __init__(self, root_dir, label_dir):
            self.root_dir = root_dir
            self.label_dir = label_dir
            self.path=os.path.join(self.root_dir,self.label_dir) #获得图片的路径地址
            self.img_path = os.listdir(self.path) #获得图片排成列表
      def __getitem__(self, idx):
            img_name = self.img_path[idx] #idx取列表里的第几张图片  idx=0
            img_item_path = os.path.join(self.root_dir, self.label_dir, img_name) #取图片的路径
            img = Image.open(img_item_path)  #读取图片
            label = self.label_dir
            return img, label

      def __len__(self):
            return  len(self.img_path)

root_dir = "dataset/train"
ants_label_dir = "ants"
ants_dataset = MyData(root_dir, ants_label_dir)
print(ants_dataset)
print(ants_dataset[0])  # 根据重写的getitem返回 img与 label
img, label = ants_dataset[0]
img.show()
bees_label_dir = "bees"
bees_dataset = MyData(root_dir, bees_label_dir)
img, label = bees_dataset[0]
img.show()

train_dataset = ants_dataset + bees_dataset  # 两个数据集的拼接 未改变顺序，ants在前 bees在后
print(len(ants_dataset))
print(len(bees_dataset))
print(len(train_dataset))

