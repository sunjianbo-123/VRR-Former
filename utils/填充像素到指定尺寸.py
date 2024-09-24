from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
imgpath = './RealData_0.jpg'
img = np.array(Image.open(imgpath))
print(img.shape)

img_0 = 720-img.shape[0]      # 第0个维度填充到200需要的像素点个数
img_1 = 1280-img.shape[1]     # 第1个维度填充到200需要的像素点个数


a = random.randint(0, img_0)
b = random.randint(0, img_1)

img_pad=np.pad(img, ((a, img_0-a), (b, img_1-b), (0, 0)), 'constant', constant_values=255)
# (10,img_0-10)表示在第0个维度上，前面填充10个像素点，后面填充img_0-10个像素点
# np.pad()的参数：（原始矩阵，填充的行数，填充的列数，填充的方式）


new_im = Image.fromarray(img_pad.astype(np.uint8))
new_im.save('./RealData_0_pad.jpg')



