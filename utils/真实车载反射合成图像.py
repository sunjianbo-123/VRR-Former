import cv2
import os
import random
from PIL import Image
import numpy as np


# 批量处理
file_transmition = '/home/boryant/download/ReflectionRemovalDatas10672/RealDatas2374/transmition_layer2374/day+night_2374'
file_reflection  = '/home/boryant/download/ReflectionRemovalDatas10672/RealDatas2374/reflection_layer2374/day+night_2374jpg'
blended_path     =  '/home/boryant/download/ReflectionRemovalDatas10672/RealDatas2374/belnded2374'
images = os.listdir(file_transmition)


for image in images:       # 遍历文件夹中文件名
    print(str(image))
    reflection_image = Image.open(file_reflection + '/' + str(image)).convert("RGBA")
    index = image.rfind('.')
    a = image[:index]
    print(str(a))
    transmition_image = Image.open(file_transmition + '/' + str(a) + ".jpg")

    datas = reflection_image.getdata()

    newData = []
    for item in datas:
        # If all RGB values are 255 (white), then set alpha to 0 (100% transparent)
        if item[0] > 240 and item[1] > 240 and item[2] > 240:
            newData.append((255, 255, 255, 0))
        else:
            # Else set alpha to 178 (30% transparent)
            newData.append((item[0], item[1], item[2], 178))

    # Update the image data
    reflection_image.putdata(newData)
    # reflection_image.save(file_reflection_1 + '/' + str(a) + ".png")

    # Ensure the reflection image isn't larger than the background
    reflection_image.thumbnail(transmition_image.size, Image.ANTIALIAS)



    # Randomly choose a position to place the reflection image on the background
    max_x = transmition_image.size[0] - reflection_image.size[0]
    max_y = transmition_image.size[1] - reflection_image.size[1]
    pos_x = random.randint(0, max_x)
    if max_y > 240:
        pos_y = random.randint(240, max_y)
    else:
        pos_y = random.randint(0, max_y)



    # Paste the reflection image onto the background image
    # 防止在使用paste方法后图片颜色溢出，需要确保reflection_image具有与transmission_image相同的模式
    # 确保 transmission_image 是 RGBA 模式
    # if transmition_image.mode != 'RGBA':
    #     transmition_image = transmition_image.convert('RGBA')
    # 将t 和 r_blur 图像转换为 RGBA 模式


    blend = transmition_image.paste(reflection_image, (pos_x, pos_y), reflection_image.split()[-1])


    # Save the combined image
    transmition_image.save(blended_path + '/' + str(a) + ".jpg")
    print("saved image to {}" .format(blended_path + '/' + str(a) + ".jpg"))








