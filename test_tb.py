#tensorboard的使用
# 把图片放到tensorboard里面显示出来
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs") # 日志文件存储位置
img_path = "data/val/ants/10308379_1b6c72e180.jpg"
img = Image.open(img_path)
print(type(img))
img_array = np.array(img)
print(type(img_array))
print(img_array.shape)
writer.add_image("Img Test", img_array, 1, dataformats="HWC")
writer.close()

