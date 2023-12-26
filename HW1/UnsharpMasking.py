import numpy as np
import cv2

def processingImg(imagePath):
    image = cv2.imread(imagePath, 0)
    # 4. 將影像轉換成 double 格式，數值範圍在[0 1]之間。
    image = image/255  # float64
    return image

def unsharpMask(image, kernel):
    # 5. 用雙層迴圈對 n x n 濾鏡(均值濾鏡或高斯濾鏡)做影像模糊化，獲得近似圖3(b)的結果。
    img_row, img_col = image.shape
    ker_row, ker_col = kernel.shape
    pad = int((ker_col - 1) / 2)
    image = np.pad(image, (pad, pad), 'symmetric')
    output = np.zeros((img_row, img_col), dtype=np.float64)
    for y in range(img_row):
        for x in range(img_col):
            cim = image[y:y + 3, x:x + 3]
            result = (cim * kernel).sum()
            output[y, x] = result
    return output

# filters
mean_filter = np.ones((3,3))/9
gaussian_Filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

# load image
raw_img = processingImg('ntust_gray.jpg')
# 5. 用雙層迴圈對 n x n 濾鏡(均值濾鏡或高斯濾鏡)做影像模糊化，獲得近似圖3(b)的結果。
blur_image = unsharpMask(raw_img, gaussian_Filter)
# 6. 利用原圖與模糊影像的差異，加上原圖，獲得類似圖 3(c)的銳利影像。
usm_image = 0.8 * (raw_img - blur_image) + raw_img  # 0.8(raw - processed) + raw

raw_img = raw_img * 255
blur_image = blur_image * 255
usm_image = usm_image * 255

cv2.imwrite("gaussian_Filter_image.jpg", blur_image)
cv2.imwrite("gaussian_usm_image.jpg", usm_image)