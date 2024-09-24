import os
from astropy.io import fits
import matplotlib.pyplot as plt


def convert_fits_to_jpg(input_folder, output_folder,dpi=300):
    # 检查输出文件夹是否存在，如果不存在就创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.fits'):
            # 构建完整的文件路径
            file_path = os.path.join(input_folder, file_name)
            # 读取FITS文件
            with fits.open(file_path) as hdul:
                # 取得FITS文件中的图像数据
                image_data = hdul[0].data




            # 创建一个图像并使用灰度色彩图
            plt.figure()
            plt.imshow(image_data, cmap='gray')
            plt.axis('off')  # 不显示坐标轴

            # 构建输出的JPG文件路径
            output_file_path = os.path.join(output_folder, file_name.replace('.fits', '.jpg'))
            # 保存图像
            plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0, dpi=dpi, format='jpg')
            plt.close()




# 调用函数
input_folder = '/home/sunjianbo/PycharmProjects/MyProjects/ReflectionRemoval/Uformer-main/datasets/sun/China/'     # 替换为你的输入文件夹路径
output_folder = '/home/sunjianbo/PycharmProjects/MyProjects/ReflectionRemoval/Uformer-main/datasets/sun/China/'    # 替换为你的输出文件夹路径
convert_fits_to_jpg(input_folder, output_folder)
