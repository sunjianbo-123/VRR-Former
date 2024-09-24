import os
import shutil

def copy_selected_images(source_folder, destination_folder):
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        # 检查文件名是否包含指定的字符串
        if any(substring in filename for substring in ["ambient_T", "ambient_R", "pureflash"]):
            # any() 函数用于判断给定的可迭代参数 iterable 是否全部为 False，则返回 False，如果有一个为 True，则返回 True
            # 构建源文件和目标文件的完整路径
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)

            # 复制文件
            shutil.copy(source_file, destination_file)
            print(f"Copied: {filename}")

# 指定源文件夹和目标文件夹
source_folder = '/home/boryant/download/ReflectionRemovalDatas10672/SyntheticData8298/ReflectionLayer8298/' \
                'flash-reflection-removal/data/synthetic/with_syn_reflection/train/others/'
destination_folder = '/home/boryant/download/ReflectionRemovalDatas10672/SyntheticData8298/ReflectionLayer8298/' \
                     'flash-reflection-removal/data/synthetic/with_syn_reflection/train_r/'

copy_selected_images(source_folder, destination_folder)
