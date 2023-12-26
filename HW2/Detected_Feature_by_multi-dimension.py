import numpy as np
import cv2
import os


def imgMask(img, threshold):
    """
    :param img:
    :param threshold:
    :return: image mask in float32, [0, 1]
    """
    img_ = img.copy()
    h, w = img_.shape[0], img_.shape[1]
    output = np.zeros([h, w]).astype(np.float32)
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    img_ = img_.astype(np.float32)
    for y in range(img_.shape[0]):
        for x in range(img_.shape[1]):
            pixel = img_[y, x]
            if pixel > threshold:
                new_pixel = 0
            else:
                new_pixel = 1
            output[y, x] = new_pixel
    return output


def maskingImg(img, threshold):
    img_ = img.copy()
    img_mask = imgMask(img_, threshold)
    img_mask = np.argwhere(img_mask != 1)
    img_[img_mask[:, 0], img_mask[:, 1]] = 0
    img_ = (img_ / 255).astype(np.float32)
    return img_


def imgCenter(img):
    img_ = img.copy()
    img_ = imgMask(img_, 250)
    h, w = img_.shape[0], img_.shape[1]
    img_area = img_.sum()
    sum_x = 0
    sum_y = 0
    for y in range(h):
        for x in range(w):
            if img_[y, x] == 1.0:
                sum_y += y
                sum_x += x
    cy = round(sum_y / img_area)
    cx = round(sum_x / img_area)
    return cy, cx


def plotCenter(img):
    img_ = img.copy()
    h, w = img_.shape[0], img_.shape[1]
    cy, cx = imgCenter(img_)
    img_masked = imgMask(img_, 250)
    img_masked = cv2.cvtColor(img_masked, cv2.COLOR_GRAY2BGR)  # 3-D
    new_img = cv2.line(img_masked, (0, cy), (w, cy), (0, 0, 1), 1)
    new_img = cv2.line(new_img, (cx, 0), (cx, h), (0, 0, 1), 1)
    return new_img


def sigCurve(img, bins=60):
    img_ = img.copy()
    img_masked = imgMask(img_, 250)
    interval = 360 / bins
    r_histogram = np.zeros([bins, 1])
    cy, cx = imgCenter(img_)
    for y in range(img_masked.shape[0]):
        for x in range(img_masked.shape[1]):
            if img_masked[y, x] == 1:
                theta = np.mod(np.arctan2(cy - y, cx - x) * 180 / np.pi, 360)
                i = int(np.ceil(theta / interval))
                i = int(np.floor(theta / interval))
                r = np.sqrt((cy - y) ** 2 + (cx - x) ** 2)
                if r > r_histogram[i, 0]:
                    r_histogram[i] = r
    # x = np.arange(r_histogram.size)
    img_gradient = abs(np.gradient(r_histogram, axis=0)).sum()
    avg_img_gradient = img_gradient / bins
    return avg_img_gradient


def avgLightness(img):
    img_ = img.copy()
    img_masked = maskingImg(img_, 250)
    img_masked = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_index = np.argwhere(img_masked != 0)
    avg = img_masked.sum() / gray_index.shape[0]
    return avg


def redRatio(img):
    img_ = img.copy()
    img_masked = maskingImg(img_, 250)
    ch_ratio = []
    for ch in img_masked[:, :, 0], img_masked[:, :, 1], img_masked[:, :, 2]:  # BGR
        channel_index = np.argwhere(ch > 0)
        ch_avg = ch.sum() / channel_index.shape[0]
        ch_ratio.append(ch_avg)
    output = ch_ratio[2] / sum(ch_ratio)
    return output


def avgHiPass(img):
    img_ = img.copy()
    mask = imgMask(img_, 250)
    mask_indexing = np.argwhere(mask != 0)
    laplac_img = cv2.Laplacian(img_, -1, ksize=3)
    laplac_img = abs(laplac_img)
    output = laplac_img.sum() / mask_indexing.shape[0]
    return output


if __name__ == "__main__":
    y = 800
    x = 800
    blank_img_lr = np.ones([y, x, 3]).astype(np.float32)
    blank_img_ls = np.ones([y, x, 3]).astype(np.float32)

    folder_dir = "C:/Users/cghsi/Desktop/HW2/leaves/"
    list1 = []
    for i in os.listdir(folder_dir):
        img = cv2.imread(folder_dir + i)
        img_float = img.copy()
        img_float = img_float / 255
        masked_img = imgMask(img, 250)
        masked_img_index = np.argwhere(masked_img != 0)
        # 平均亮度
        img_avg_lightness = avgLightness(img)
        # 紅色平均占比
        img_red_ratio = redRatio(img)
        # 樹葉特徵分布圖1的座標
        by = np.floor(img_avg_lightness * y).astype(np.int32)
        bx = np.floor(img_red_ratio * x).astype(np.int32)
        # 平均高頻強度
        img_avg_lap = avgHiPass(img) / 400
        # 簽名曲線的梯度平均
        img_avg_sig = sigCurve(img) / 6.120549287353109
        # 樹葉特徵分布圖2的座標
        cy = np.floor(img_avg_lap * y).astype(np.int32)
        cx = np.floor(img_avg_sig * x).astype(np.int32)
        # 樹葉特徵分布圖1
        blank_img_lr[masked_img_index[:, 0] + (y - by), masked_img_index[:, 1] + bx] = img_float[masked_img_index[:, 0],
                                                                                                 masked_img_index[:, 1]]
        # 樹葉特徵分布圖2
        blank_img_ls[masked_img_index[:, 0] + (y - cy), masked_img_index[:, 1] + cx] = img_float[masked_img_index[:, 0],
                                                                                                 masked_img_index[:, 1]]

    blank_img_ls = (blank_img_ls * 255).astype(np.uint8)
    blank_img_lr = (blank_img_lr * 255).astype(np.uint8)

    cv2.imwrite("hw3/blank_img_ls.jpg", blank_img_ls)
    cv2.imwrite("hw3/blank_img_lr.jpg", blank_img_lr)
    # cv2.imshow("a", blank_img_lr)
    # cv2.imshow("b", blank_img_ls)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows
