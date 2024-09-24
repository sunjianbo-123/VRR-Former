import os
import re
import shutil


def extract_number(filename):
    # 使用正则表达式从文件名中提取数字
    numbers = re.findall(r'\d+', filename)
    # "\d+" 正则表达式匹配数字，"\d"匹配数字
    return int(numbers[0]) if numbers else 0


def copy_first_n_images(source_folder, destination_folder, n):
    # 获取源文件夹中的所有文件名
    files = os.listdir(source_folder)
    # 过滤出图片文件，并根据文件名中的数字排序
    image_files = sorted([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))], key=extract_number)
    # sorted() 函数对所有可迭代的对象进行排序操作   key 指定比较的对象
    # key=extract_number 指定比较的对象为extract_number函数返回的值



    # 复制前n张图片
    for filename in image_files[:+n]:
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        shutil.copy(source_file, destination_file)
        print(f"Copied: {filename}")


# 指定源文件夹和目标文件夹路径
source_folder = '/home/boryant/download/ReflectionRemovalDatas/SyntheticData10693/blended10693/'
destination_folder = '/home/boryant/download/ReflectionRemovalDatas/SyntheticData10693/blended10693/total/'
number_of_images_to_copy = 10693 # 你想复制的图片数量

copy_first_n_images(source_folder, destination_folder, number_of_images_to_copy)
