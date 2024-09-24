from __future__ import division
import os, cv2
import numpy as np
import scipy.stats as st


# Adjust the paths according to your directory structure

train_r_path = "/home/boryant/download/ReflectionRemovalDatas10672/RealDatas2374/reflection_layer2374/day+night_2374jpg/"
train_r_blur_path = "/home/boryant/download/ReflectionRemovalDatas10672/RealDatas2374/reflection_layer2374/day+night_2374jpg_blur/"



IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def gkern(kernlen_x=1280, kernlen_y=720, nsig=1):
    """Returns a 2D Gaussian kernel array."""
    interval_x = (2 * nsig + 1.) / (kernlen_x)
    x = np.linspace(-nsig - interval_x / 2., nsig + interval_x / 2., kernlen_x + 1)
    kern1d_x = np.diff(st.norm.cdf(x))

    interval_y = (2 * nsig + 1.) / (kernlen_y)
    y = np.linspace(-nsig - interval_y / 2., nsig + interval_y / 2., kernlen_y + 1)
    kern1d_y = np.diff(st.norm.cdf(y))

    kernel_raw = np.sqrt(np.outer(kern1d_y, kern1d_x))
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel / kernel.max()
    return kernel




def syn_data( r, train_r_blur, sigma, index):
    # 对透射层（t）和反射层（r）图像进行伽马校正，增加对比度，使得颜色更饱和

    r = np.power(r, 2.2)

    # 计算高斯核的大小
    sz = int(2 * np.ceil(2 * sigma) + 1)
    # np.ceil(x)  大于等于x的最小整数

    # 对反射层图像应用高斯模糊，模拟真实世界中反射的模糊效果
    r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
    # sz必须是奇数 sigma X Y方向上的高斯核标准偏差





    att = 1.08 + np.random.random() / 10.0

    # 通过对混合图像的每个颜色通道进行操作，调整亮度和对比度，以确保图像的亮度在合理范围内
    r_blur[r_blur >= 1] = 1
    r_blur[r_blur <= 0] = 0



    # 应用渐晕遮罩和透明度调整：
    # 创建一个局部渐晕遮罩（alpha1）和一个透明度因子（alpha2），并将其应用到混合图像上，增加了反射层的不均匀透明度。
    h, w = r_blur.shape[0: 2]

    if 1280 - w - 10 > 0:
        neww = np.random.randint(0, 1280 - w - 10)
    else:
        neww = 0  # 或者其他合适的默认值
    if 720 - h - 10 > 0:
        newh = np.random.randint(0, 720 - h - 10)
    else:
        newh = 0
    # neww = np.random.randint(0, 1280 - w - 10)
    # newh = np.random.randint(0, 720 - h - 10)

    # create a vignetting mask
    # 创建一个渐晕遮罩，这个遮罩会被应用到反射层图像上，以模拟光线在图像边缘处的自然衰减效果

    g_mask = gkern(w, h, 3)
    g_mask = np.dstack((g_mask, g_mask, g_mask))

    alpha1 = g_mask[0: h, 0: w, :]

    r_blur = np.multiply(r_blur, alpha1)


    # 通过对反射层应用逆伽马校正，将图像的颜色空间恢复到原始状态
    r_blur = np.power(np.multiply(r_blur, alpha1), 1 / 2.2)

    cv2.imwrite(os.path.join(train_r_blur, "RealData_{}.jpg".format(str(index))),
                cv2.normalize(r_blur, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F))
    print("saved reflection{}.jpg".format(str(index)))

    return r_blur



def prepare_data( train_r_path):

    r_images = [os.path.join(train_r_path, f) for f in os.listdir(train_r_path) if is_image_file(f)]
    return  r_images



if True:
    r_images_list = prepare_data(train_r_path)

    for r_path in r_images_list:

        r_img = cv2.imread(r_path, -1)
        image_index = r_path.split('/')[-1].split('.')[0].split('_')[-1]






        # Convert images to float and normalize
        r_img = np.float32(r_img) / 255.0

        # Randomly select sigma value for Gaussian blur
        k_sz = np.linspace(1, 5, 80)   # for synthetic images
        sigma = k_sz[np.random.randint(0, len(k_sz))]

        syn_data(r_img, train_r_blur_path, sigma, image_index)