import cv2
import numpy as np
import PIL.Image as Image
import os

def img_rotate(src, angel):
    """逆时针旋转图像任意角度

    Args:
        src (np.array): [原始图像]
        angel (int): [逆时针旋转的角度]

    Returns:
        [array]: [旋转后的图像]
    """
    h,w = src.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angel, 1.0)
    # 调整旋转后的图像长宽
    rotated_h = int((w * np.abs(M[0,1]) + (h * np.abs(M[0,0]))))
    rotated_w = int((h * np.abs(M[0,1]) + (w * np.abs(M[0,0]))))
    M[0,2] += (rotated_w - w) // 2
    M[1,2] += (rotated_h - h) // 2
    # 旋转图像
    rotated_img = cv2.warpAffine(src, M, (rotated_w,rotated_h))

    return rotated_img


def rotate_bound(image, angle):
    """

    :param image: 原图像
    :param angle: 旋转角度
    :return: 旋转后的图像
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    img = cv2.warpAffine(image, M, (nW, nH))
    # perform the actual rotation and return the image
    return img



def fixed_size(file, width=1024, height=1024):
    """按照固定尺寸处理图片"""
    im = Image.open(file)
    out = im.resize((width, height), Image.ANTIALIAS)
    #Image.ANTIALTAS的作用是
    out.save(file)





if __name__ == "__main__":
    # 读取图像
    img_path = "/home/boryant/download/X-EUV images/X-EUV images/111提取网格/20230217-22_15_16/原始图像/20230217-22_15_16-800ms#-1675.080##1464.9377###6844.1894-solar_grey_data_aligned_0.jpg"
    img = cv2.imread(img_path)
    for i in range(0, 360):
        img_rotated = img_rotate(img, i)
        # 改成和原图一样的名字
        cv2.imwrite("/home/boryant/download/X-EUV images/transmition/20230217-22_15_16-800ms#-1675.080##1464.9377###6844.1894-solar_grey_data_{}.jpg".format(i), img_rotated)
        print("saved {} images".format(i+1))


