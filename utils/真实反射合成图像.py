from __future__ import division
import os, cv2
import numpy as np
import scipy.stats as st


# Adjust the paths according to your directory structure
train_t_path = "/home/boryant/desktop/T/"
train_r_path = "/home/boryant/desktop/R/"
train_b_path = "/home/boryant/desktop/B/"



IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# functions for synthesizing images with reflection
# def gkern(kernlen=100, nsig=1):
#     """Returns a 2D Gaussian kernel array."""
#     interval = (2 * nsig + 1.) / (kernlen)
#     x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
#     kern1d = np.diff(st.norm.cdf(x))
#     kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
#     kernel = kernel_raw / kernel_raw.sum()
#     kernel = kernel / kernel.max()
#     return kernel


# 生成一个高斯核，该函数返回一个二维高斯核数组，用于后续创建渐晕效果（vignetting）的遮罩
# gkern函数现在接受两个参数kernlen_x和kernlen_y，分别代表核在水平和垂直方向上的长度
# 然后，它分别生成两个一维高斯核，并通过外积来创建一个二维核。最后，这个核被归一化，并扩展为一个三通道数组
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




def syn_data(t, r, train_b, index):
    # 对透射层（t）和反射层（r）图像进行伽马校正，增加对比度，使得颜色更饱和
    t = np.power(t, 2.2)
    r = np.power(r, 2.2)


    r_blur = r
    # sz必须是奇数 sigma X Y方向上的高斯核标准偏差

    # 将模糊后的反射层与透射层相加，创建初步的反射效果
    blend = r_blur + t



    att = 1.08 + np.random.random() / 10.0

    # 通过对混合图像的每个颜色通道进行操作，调整亮度和对比度，以确保图像的亮度在合理范围内
    for i in range(3):
        maski = blend[:, :, i] > 1
        mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
        r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
    r_blur[r_blur >= 1] = 1
    r_blur[r_blur <= 0] = 0




    alpha2 = 1 - np.random.random() / 5.0
    blend = r_blur + t * alpha2


    # 通过对混合图像、透射层、反射层应用逆伽马校正，将图像的颜色空间恢复到原始状态
    t = np.power(t, 1 / 2.2)
    r_blur_mask = np.power(r_blur, 1 / 2.2)
    blend = np.power(blend, 1 / 2.2)
    blend[blend >= 1] = 1
    blend[blend <= 0] = 0


    cv2.imwrite(os.path.join(train_b, "SynData_{}.jpg".format(str(index))),
                cv2.normalize(blend, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F))
    print("saved blended{}.jpg".format(str(index)))

    return blend



def prepare_data(train_t_path, train_r_path):
    t_images = [os.path.join(train_t_path, f) for f in os.listdir(train_t_path) if is_image_file(f)]
    r_images = [os.path.join(train_r_path, f) for f in os.listdir(train_r_path) if is_image_file(f)]
    return t_images, r_images



if True:
    t_images_list, r_images_list = prepare_data(train_t_path, train_r_path)

    for (t_path, r_path) in zip(t_images_list, r_images_list):
        t_img = cv2.imread(t_path, -1)
        r_img = cv2.imread(r_path, -1)
        image_index = t_path.split('/')[-1].split('.')[0].split('_')[-1]



        # Resize reflection image to match transmission image (1280x720)
        """
        cv2.resize() 函数使用了 INTER_AREA 插值方法，这通常用于缩小图片时保持较好的质量
        如果反射层图片需要被放大，可能需要选择其他插值方法，如 cv2.INTER_LINEAR 或 cv2.INTER_CUBIC
        """
        if r_img.shape[0] > 720 and r_img.shape[1] > 1280:
            r_img_resized = cv2.resize(r_img, (1280, 720), interpolation=cv2.INTER_AREA)
        else:
            r_img_resized = cv2.resize(r_img, (1280, 720), interpolation=cv2.INTER_CUBIC)


        # Convert images to float and normalize
        t_img = np.float32(t_img) / 255.0
        r_img_resized = np.float32(r_img_resized) / 255.0



        syn_data(t_img, r_img_resized, train_b_path, image_index)














