from __future__ import division
import os, cv2
import numpy as np
import scipy.stats as st
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task",  default="experiment_with_bn+relu", help="path to folder containing the model")
parser.add_argument("--data_syn_dir",  default=" ", help="path to synthetic dataset")
parser.add_argument("--data_real_dir",  default=" ", help="path to real dataset")
parser.add_argument("--save_model_freq",  default=1, type=int, help="frequency to save model")
parser.add_argument("--is_hyper",  default=1, type=int, help="use hypercolumn or not")
parser.add_argument("--is_training",  default=1, help="training or testing")
parser.add_argument("--continue_training",  action="store_true", help="search for checkpoint in the subfolder specified by `task` argument")
ARGS = parser.parse_args()



train_syn_root = [ARGS.data_syn_dir]
train_real_root = [ARGS.data_real_dir]




IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



# functions for synthesizing images with reflection (details in the paper)
def gkern(kernlen=100, nsig=1):
    """Returns a 2D Gaussian kernel array."""
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel / kernel.max()
    return kernel


# create a vignetting mask
g_mask = gkern(720, 3)                               # (560,3)
g_mask = np.dstack((g_mask, g_mask, g_mask))




def syn_data(t, r, index):
    alpha2 = 1 - np.random.random() / 5.0      # np.random.random()生成0-1之间的随机数  最终产生0.8-1之间的随机数
    blend = np.multiply(r, 1) + t * alpha2     # np.multiply()对应元素相乘

    # # 创建一个随机的W
    # W = np.random.random((r.shape[0], r.shape[1], 1))
    # blend = r*W + t * (1.0 - W)








    # cv2.imwrite("/home/sunjianbo/PycharmProjects/MyProjects/ReflectionRemoval/perceptual-reflection-removal/training_synthetic_data/transmission_layer/{}.jpg".format(str(index)),
    #             cv2.normalize(t, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F))
    # print("saved {}{}.jpg".format("transmission", str(index)))

    print("saved {}{}.jpg".format("reflection", str(index)))

    cv2.imwrite("/home/boryant/PycharmProjects/MyProjects/ReflectionRemovalCodes/perceptual-reflection-removal/training_synthetic_data/B/{}.jpg".format(str(index)),
                cv2.normalize(blend, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F))
    print("saved {}{}.jpg".format("blended", str(index)))
    return  blend



if True:
    # please follow the dataset directory setup in README
    def prepare_data(train_path):
        input_names = []
        image1 = []
        image2 = []
        for dirname in train_path:
            train_t_gt = dirname + "T/"
            train_r_gt = dirname + "R/"
            train_b = dirname + "B/"
            for root, _, fnames in sorted(os.walk(train_t_gt)):
                for fname in fnames:
                    if is_image_file(fname):
                        path_input = os.path.join(train_b, fname)
                        path_output1 = os.path.join(train_t_gt, fname)
                        path_output2 = os.path.join(train_r_gt, fname)
                        input_names.append(path_input)
                        image1.append(path_output1)
                        image2.append(path_output2)
        return input_names, image1, image2


    _, syn_image1_list, syn_image2_list = prepare_data(train_syn_root)


    for id in range(1,16):

        syn_image1 = cv2.imread(syn_image1_list[id], -1)
        # neww = np.random.randint(256, 480)
        neww = 640
        newh = round((neww / syn_image1.shape[1]) * syn_image1.shape[0])

        syn_image2 = cv2.imread(syn_image2_list[id], -1)
        output_image_t = cv2.resize(np.float32(syn_image1), (neww, newh), cv2.INTER_CUBIC) / 255.0
        output_image_r = cv2.resize(np.float32(syn_image2), (neww, newh), cv2.INTER_CUBIC) / 255.0
        file = os.path.splitext(os.path.basename(syn_image1_list[id]))[0]

        # if np.mean(output_image_t) * 1 / 2 > np.mean(output_image_r):
        #     continue
        _ = syn_data(output_image_t, output_image_r, file)
