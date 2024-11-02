from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms
#transformsd中的toTensor的使用：
# 把pil和mupy转化成tensor类型
writer = SummaryWriter("logs") # 日志文件存储位置
img_path = "dataset/train/ants/5650366_e22b7e1065.jpg"
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer.add_image("Tensor_img",tensor_img)
writer.close()#关闭
#normalize
#创建一个归一化转换，将图像的每个通道按照指定的均值和标准差进行归一化。
#将归一化前后的像素值打印出来以便于观察。
#将归一化后的图像保存到TensorBoard中以便于后续的可视化分析。
trans_norm = transforms.Normalize([85, 5, 7], [8, 5, 30])
img_norm = trans_norm(tensor_img)
print(tensor_img[0][0][0])
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm,2)
writer.close()
#resize
img = Image.open(img_path)
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
print("对PIL类型进行Resize",img_resize)
tensor_trans = transforms.ToTensor()
img_resize = tensor_trans(img_resize)
writer.add_image("Resize",img_resize,0)
print(img_resize)
#随机裁剪RandomCrop
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms

writer = SummaryWriter("logs")  # 日志文件存储位置
img = Image.open(img_path)
print(img)
trans_random = transforms.RandomCrop(30)
trans_totensor = transforms.ToTensor()
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()


