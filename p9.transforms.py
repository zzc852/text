#transforms的使用：是对图片进行一些变换

from PIL import  Image
from torchvision import  transforms

#python的用法---》tensor数据类型
#1.transforms该如何使用（python）
#2.为什么我们需要tensor数据类型
#包装了神经网络所需要的理论基础参数
img_path="dataset/train/ants/0013035.jpg"
img=Image.open(img_path)
#1.transforms该如何使用（python）
tensor_teans=transforms.ToTensor()
tensor_img=tensor_teans(img)
print(tensor_img)