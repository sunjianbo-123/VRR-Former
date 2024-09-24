import os
import cv2
import numpy as np
from PIL import Image

filepath = "/home/boryant/download/ReflectionRemovalDatas10672/RealDatas2374/belnded2374/"
filename = os.listdir(filepath)
base_dir = filepath
new_dir = "/home/boryant/download/ReflectionRemovalDatas10672/RealDatas2374/belnded2374-1/"


i = 0
for img in filename:
    '''修改图像后缀名'''
    if os.path.splitext(img)[1] == '.png' or '.PNG':
        name = os.path.splitext(img)[0]
        newFileName = str(name) + ".jpg"
        try:
            im = Image.open(base_dir + img) # Image打开图片的通道顺序是RGB  但是cv2保存图片的通道顺序是RGB
            # im_gray1 = np.array(im)
            im.save(new_dir + newFileName)
            # cv2.imwrite(new_dir + newFileName, im_gray1)
            print('converting %s to %s ...' % (img, newFileName))
            i = i + 1
        except:
          continue




