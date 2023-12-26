import numpy as np
import cv2

__lapFil__ = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])


def imgProc(img):
    """
    讀取影像
    """
    img = img / 255
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def laplacian_filter(img):
    """
    使用Laplacian濾鏡進行影像銳利化
    """
    img = imgProc(img)
    h, w = img.shape
    kh, kw = __lapFil__.shape
    pad = int((kh - 1) / 2)
    pad_img = np.pad(img, (pad, pad), "symmetric")
    output = np.zeros([h, w], dtype=np.float32)
    for y in range(h):
        for x in range(w):
            ci = pad_img[y:y+3, x:x+3]  # ci is crop image
            result = (ci * __lapFil__).sum()
            output[y, x] = result
    return abs(output) * 3


def median_filter(img):
    """
    均值綠波濾鏡使影像不破碎化
    """
    mean_kernel = np.ones([23, 23]) / 23**23
    h, w = img.shape
    kh, kw = mean_kernel.shape
    pad = int((kh - 1) / 2)
    pad_img = np.pad(img, (pad, pad), "symmetric")
    output = np.zeros([h, w], dtype=np.float32)
    for y in range(h):
        for x in range(w):
            cp = pad_img[y:y+23, x:x+23]
            result = (cp * mean_kernel).sum()
            output[y, x] = result
    return output


def imgTreshold(img):
    """
    二值化函數
    """
    h, w = img.shape
    output = np.zeros([h, w], dtype=np.float32)
    for y in range(h):
        for x in range(w):
            pixel = img[y, x]
            if pixel > 0:
                pixel = 1
            else:
                pixel = 0
            output[y, x] = pixel
    return output


def cvtDtype(img):
    output = (img * 255).astype(np.uint8)
    return output


if __name__ == "__main__":
    # 2. 讀取並顯示對焦在前景(fg)與背景(bg)的兩幅影像，並轉換至float格式。
    fgimg = cv2.imread('./depthOfField/2fg.jpg')
    bgimg = cv2.imread('./depthOfField/2bg.jpg')
    # 4. 高通濾波：將兩幅影像由彩色轉換至灰階格式，並分別做 Laplacian 高通濾波後，取絕對值。
    fg_hipass = laplacian_filter(fgimg)
    bg_hipass = laplacian_filter(bgimg)
    # 6. 製作前景遮罩mask = fg_hipass - bg_hipass
    mask = fg_hipass - bg_hipass
    # 7. 將遮罩做「均值濾波」，濾鏡尺寸要很大，才不至於使區塊破碎。
    mask = median_filter(mask)
    # 8. 以0為門檻，將前景遮罩二值化。
    img_tresh = imgTreshold(mask)
    # 10. 根據二值遮罩分別取前景(fg)與背景(bg)的清晰像素，組成景深擴增影像。
    img_tresh_index = np.argwhere(img_tresh > 0)
    new_img = bgimg.copy()
    new_img[img_tresh_index[:, 0], img_tresh_index[:, 1]] = fgimg[img_tresh_index[:, 0], img_tresh_index[:, 1]]
    # 12. 儲存景深擴增影像。
    fg_hipass = cvtDtype(fg_hipass)
    bg_hipass = cvtDtype(bg_hipass)
    img_tresh = cvtDtype(img_tresh)
    img_tresh = cv2.cvtColor(img_tresh, cv2.COLOR_GRAY2BGR)
    print(img_tresh.shape)


    cv2.imwrite("hw1/fg_hipass.jpg", fg_hipass)
    cv2.imwrite("hw1/bg_hipass.jpg", bg_hipass)
    cv2.imwrite("hw1/img_tresh.jpg", img_tresh)
    cv2.imwrite("hw1/new_img.jpg", new_img)
    # cv2.imshow("a", new_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows
