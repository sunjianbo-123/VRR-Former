# coding=utf-8
import PIL.Image as Image
import os
import cv2

PHOTO_FILE = '/home/sunjianbo/PycharmProjects/MyProjects/ReflectionRemoval/Uformer-main/datasets/reflectionremoval/my_syn/test/groundtruth/'

def fixed_size(file):
    """按照固定尺寸处理图片"""
    # im = Image.open(file)
    # out = im.resize((width, height), Image.ANTIALIAS)
    img = cv2.imread(file)
    # out = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_CUBIC)
    out = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    """
    cv2.resize() 函数使用了 INTER_AREA 插值方法，这通常用于缩小图片时保持较好的质量
    如果反射层图片需要被放大，可能需要选择其他插值方法，如 cv2.INTER_LINEAR 或 cv2.INTER_CUBIC
    """

    cv2.imwrite(file, out)

    # out.save(file)

def executeCompressImage():      #  执行图片的缩放
    for r, d, f in os.walk(PHOTO_FILE):
        for file in f:
           if '.jpg' or ".png" in file:
               path = os.path.join(r, file)
               # print(path)
               fixed_size(path)
               print(path)

executeCompressImage()





