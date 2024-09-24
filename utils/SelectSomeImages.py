import os, random, shutil


def moveFile(fileDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    # rate = 1
    # picknumber = int(filenumber * rate)          # 按照rate比例从文件夹中取一定数量图片
    picknumber = 3000
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(sample)
    for name in sample:
        shutil.copy(fileDir + name, tarDir + name)
    print('total %d to copy' % (picknumber))
    return


if __name__ == '__main__':
    fileDir = "/home/boryant/Pictures/synthetic/transmission_layer/"                #  源图片文件夹路径
    tarDir = './training_synthetic_data/transmission_layer/'                      #  移动到新的文件夹路径
    moveFile(fileDir)