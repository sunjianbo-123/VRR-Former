from PIL import Image
from skimage import exposure, img_as_float, io  # 导入所需要的 skimage 库
import os

old_path = "../change_light_jpg/"  # 原始文件路径
save_path = "./"  # 需要存储的文件路径

file_list = os.walk(old_path)

for root, dirs, files in file_list:
    for file in files:
        pic_path = os.path.join(root, file)  # 每一个图片的绝对路径
        # 读取图像
        img_org = Image.open(pic_path)
        # 转换为 skimage 可操作的格式
        img = img_as_float(img_org)

        # 调整图像亮度，数值低于1.0，表示调亮；高于1.0表示调暗。
        img_light = exposure.adjust_gamma(img, 0.7)
        img_dark = exposure.adjust_gamma(img, 1.5)

        # 存储文件到新的路径中，并修改文件名
        io.imsave(os.path.join(save_path, file[:-4] + "-light.jpg"), img_light)
        io.imsave(os.path.join(save_path, file[:-4] + "-dark.jpg"), img_dark)

