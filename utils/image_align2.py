import cv2
import numpy as np

def align_images(im1, im2):
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(2000)

    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches1 = matcher.knnMatch(descriptors1, descriptors2, k=2)
    matches2 = matcher.knnMatch(descriptors2, descriptors1, k=2)

    # 应用 Lowe 的比率测试
    ratio_thresh = 0.8
    good_matches = []
    for m, n in matches1:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # 应用双向匹配
    final_matches = []
    for match1 in good_matches:
        for m, n in matches2:
            if match1.queryIdx == n.trainIdx and match1.trainIdx == m.queryIdx:
                final_matches.append(match1)
                break

    if len(final_matches) < 4:
        print("Not enough good matches are found - %d/%d" % (len(final_matches), 4))
        return None

    points1 = np.zeros((len(final_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(final_matches), 2), dtype=np.float32)
    for i, match in enumerate(final_matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    if h is None:
        print("Homography could not be computed.")
        return None

    height, width = im2.shape[:2]
    im1_aligned = cv2.warpPerspective(im1, h, (width, height))
    return im1_aligned



# 读取图像
im1 = cv2.imread('/home/sunjianbo/PycharmProjects/MyProjects/ReflectionRemoval/Uformer-main/datasets/sun/China/UTC20240417_040322_曝光时间1.6s_EUV1通道_单通道成像模式.jpg')   # 模糊图像路径
im2 = cv2.imread('/home/sunjianbo/PycharmProjects/MyProjects/ReflectionRemoval/Uformer-main/datasets/sun/America/AIA20240417_033600_0193.fi-new.jpg')                     # 清晰图像路径

# 对齐图像
aligned_image = align_images(im1, im2)

# 保存对齐后的图像
cv2.imwrite('/home/sunjianbo/PycharmProjects/MyProjects/ReflectionRemoval/Uformer-main/datasets/sun/China/aligned_image.jpg', aligned_image)
print("Aligned image saved as 'aligned_image.jpg'")
