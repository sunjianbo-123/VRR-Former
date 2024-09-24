from PIL import Image
from tqdm import tqdm
import glob
import os
import cv2
import numpy as np

nir_dir = '/home/boryant/download/predict/predict/'
fake_dir = '/home/boryant/download/predict/groundtruth/'
pix_dir = "/home/boryant/download/predict/subtract/"
os.makedirs(pix_dir, exist_ok=True)
files = sorted(glob.glob(os.path.join(nir_dir) + "/*.PNG"))
print(files)

for file in tqdm(files):
    nir_image = cv2.imread(file)
    print(nir_image.shape)
    fake_image = cv2.imread(fake_dir + file.split('/')[-1].replace('PNG', 'jpg'))
    print(fake_image.shape)
    image3 = cv2.subtract(nir_image, fake_image)
    # final_matrix = np.zeros((256, 256*3, 3), np.uint8)
    # final_matrix[0:256, 0:256] = nir_image
    # final_matrix[0:256, 256:512] = fake_image
    # final_matrix[0:256, 512:768] = image3
    cv2.imwrite(pix_dir + file.split('/')[-1], image3)