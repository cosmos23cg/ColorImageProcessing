import numpy as np
import cv2

def processingImg(imagePath):
    image = cv2.imread(imagePath, 0)
    # 4. 將影像轉換成 double 格式，數值範圍在[0 1]之間。
    image = image / 255  # float64

    return image

def edgeDetect(image, kernel):
    # 4. 用雙層迴圈由左而右，由上而下讀取以(x,y)為中心的 3×3 影像區域。
    # 5. 將 3×3 影像區域點對點乘上圖 1 Sobel 濾鏡數值矩陣後，將數值總和存入輸出影像的(x,y) 位置。
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

def embossing(image):
    # 6. 將濾波後的影像加上 0.5，呈現近似圖 2(b)的浮雕影像。
    image = image + 0.5
    return image

def threshold(image, threshold):
    # 7. 分別將濾波後的影像開絕對值，再二值化(門檻值自訂)，用 bitor (bitwise or)或直接相加，產生近似圖 2(c)的輪廓影像。
    # image = abs(image)
    h, w = image.shape
    img_thres = np.zeros((h, w))
    for y in range(0, h):
        for x in range(0, w):
            pixel = image[y, x]
            if pixel < threshold:
                np_pix = 0
            else:
                np_pix = 1
            img_thres[y, x] = np_pix
    return img_thres
'''
    image_set = [image1, image2]
    for item in image_set:
        print(item)
        image = abs(item)
        h, w = item.shape
        img_thres = np.zeros((h, w))
        for y in range(0, h):
            for x in range(0, w):
                pixel = image[y, x]
                if pixel < threshold:
                    np_pix = 0
                else:
                    np_pix = 1
                img_thres[y, x] = np_pix
                img_thres += img_thres
    return img_thres
'''

# filters
sobel_hor = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobel_ver = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# load image
raw_img = processingImg('ntust_gray.jpg')

# 4. 用雙層迴圈由左而右，由上而下讀取以(x,y)為中心的 3×3 影像區域。
# 5. 將 3×3 影像區域點對點乘上圖 1 Sobel 濾鏡數值矩陣後，將數值總和存入輸出影像的(x,y) 位置。
sobel_hor_img = edgeDetect(raw_img, sobel_hor)
sobel_ver_img = edgeDetect(raw_img, sobel_ver)

# 6. 將濾波後的影像加上 0.5，呈現近似圖 2(b)的浮雕影像。
sobel_hor_emb_img = embossing(sobel_hor_img)
sobel_ver_emb_img = embossing(sobel_ver_img)

# 7. 分別將濾波後的影像開絕對值，再二值化(門檻值自訂)，用 bitor (bitwise or)或直接相加，產生近似圖 2(c)的輪廓影像。
sobel_hor_thres_img = threshold(sobel_hor_img, 0.3)
sobel_ver_thres_img = threshold(sobel_ver_img, 0.3)
doubel_sobel_img = sobel_hor_thres_img + sobel_ver_thres_img
# doubel_sobel_img = (sobel_hor_img, sobel_ver_img, 0.7)

sobel_hor_emb_img = sobel_hor_emb_img * 255
sobel_ver_emb_img =sobel_ver_emb_img * 255
doubel_sobel_img = doubel_sobel_img * 255

# cv2.imshow("sobel_hor_img", doubel_sobel_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("sobel_hor_emb_img.jpg", sobel_hor_emb_img)
cv2.imwrite("sobel_ver_emb_img.jpg", sobel_ver_emb_img)
cv2.imwrite("doubel_sobel_img.jpg", doubel_sobel_img)
