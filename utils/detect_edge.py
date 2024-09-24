import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('./22.jpg', 0)

res = cv2.Canny(img, 0, 200)

# res = cv.convertScaleAbs(res) Canny边缘检测是一种二值检测，不需要转换格式这一个步骤

plt.figure(figsize=(20, 20))
plt.subplot(1, 2, 1)
m1 = plt.imshow(img, cmap=plt.cm.gray)
plt.title("原图")
plt.subplot(1, 2, 2)
m2 = plt.imshow(res, cmap=plt.cm.gray)
plt.title("Canny边缘检测")
plt.show()