import cv2
import numpy as np

M = np.array([[0.412453, 0.357580, 0.180423],  # Matrix of RGB to XYZ
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]])


def __f__(img_XYZ):
    return np.power(img_XYZ, 1 / 3) if img_XYZ > 0.00856 else (7.787 * img_XYZ) / (16 / 116)


def __anti_f__(img_XYZ):
    return np.power(img_XYZ, 3) if img_XYZ > (6 / 29) else 3 * ((6 / 29) ** 2) * (img_XYZ - 4 / 29)


# region of RGB to Lab
def __BGR2XYZ__(pixel):
    b, g, r = pixel[0], pixel[1], pixel[2]
    rgb = np.array([r, g, b])  # The order list is BGR via opencv. So I transform the order form BGR to RBG
    XYZ = np.dot(M, rgb.T)
    XYZ = XYZ / 255  # normalize
    return XYZ[0] / 0.950456, XYZ[1] / 1, XYZ[2] / 1.088754


def __XYZ2Lab__(XYZ):
    F_XYZ = [__f__(X) for X in XYZ]
    L = 116 * F_XYZ[1] - 16 if XYZ[1] > 0.00856 else 903.3 * XYZ[1]
    a = 500 * (F_XYZ[0] - F_XYZ[1])
    b = 200 * (F_XYZ[1] - F_XYZ[2])
    return L, a, b


def BGR2Lab(img):
    h = img.shape[0]
    w = img.shape[1]
    Lab = np.zeros([h, w, 3])
    for y in range(h):
        for x in range(w):
            XYZ = __BGR2XYZ__(img[y, x])
            result = __XYZ2Lab__(XYZ)
            Lab[y, x] = result[0], result[1], result[2]
    return Lab
# end region

# region of Lab to BGR
def __Lab2XYZ__(Lab):
    fY = (Lab[0] + 16.0) / 116.0
    fX = (Lab[1] / 500.0) + fY
    fZ = fY - Lab[2] / 200.0

    X = __anti_f__(fX)
    Y = __anti_f__(fY)
    Z = __anti_f__(fZ)

    X = X * 0.95047
    Y = Y * 1
    Z = Z * 1.0883
    return X, Y, Z


def __XYZ2RGB__(XYZ):
    XYZ = np.array(XYZ)
    XYZ = XYZ * 255
    rgb = np.dot(np.linalg.inv(M), XYZ.T)
    rgb = np.uint8(np.clip(rgb, 0, 255))
    return rgb


def Lab2BGR(img):
    h = img.shape[0]
    w = img.shape[1]
    new_img = np.zeros([h, w, 3])
    for y in range(h):
        for x in range(w):
            XYZ = __Lab2XYZ__(img[y, x])
            RGB = __XYZ2RGB__(XYZ)
            new_img[y, x] = RGB[2], RGB[1], RGB[0]
    new_img = new_img.astype(np.uint8)
    return new_img
# end region

def rg_blind(Lab):
    h = Lab.shape[0]
    w = Lab.shape[1]
    rg_img = np.zeros([h, w, 3])
    for y in range(h):
        for x in range(w):
            pixel = Lab[y, x]
            if pixel[1] != 0:
                pixel[1] = 0
            else:
                pixel[1] = pixel[1]
            rg_img[y, x] = (pixel[0], pixel[1], pixel[2])
    return rg_img


def yb_blind(Lab):
    h = Lab.shape[0]
    w = Lab.shape[1]
    yb_img = np.zeros([h, w, 3])
    for y in range(h):
        for x in range(w):
            pixel = Lab[y, x]
            if pixel[2] != 0:
                pixel[2] = 0
            else:
                pixel[2] = pixel[2]
            yb_img[y, x] = (pixel[0], pixel[1], pixel[2])
    return yb_img


def matlab_style_gauss2D(shape=(5, 5), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def glaucoma(img):
    h = img.shape[0]
    w = img.shape[1]
    fil = matlab_style_gauss2D(shape=(h, w), sigma=100)
    fil = fil / np.nanmax(fil)
    b, g, r = cv2.split(img)
    b = np.multiply(b, fil)
    g = np.multiply(g, fil)
    r = np.multiply(r, fil)
    result = cv2.merge([b, g, r])
    result = result.astype(np.uint8)
    return result


def rgBlindSim(img):
    lab = BGR2Lab(img)
    rg = rg_blind(lab)
    new_img = Lab2BGR(rg)
    return new_img


def ybBlindSim(img):
    lab = BGR2Lab(img)
    yb = yb_blind(lab)
    new_img = Lab2BGR(yb)
    return new_img


if __name__ == "__main__":
    img = cv2.imread('cry1.jpg')
    # 1. 紅綠色盲：自行找一張色彩豐富的圖片，將RGB影像轉換至浮點格式，再轉換至LAB空間，將a*設為0，再轉回RGB空間。
    rg_blind_img = rgBlindSim(img)
    # 2. 黃藍色盲：將RGB影像轉換至浮點格式，再轉換至LAB空間，將b*設為0，再轉回RGB空間。
    yb_blind_img = ybBlindSim(img)
    # 3. 青光眼：讀取RGB 影像的尺寸，利用fspecial 函式建立與影像同尺寸的2D高斯濾鏡(Gaussain filter)，sigma 值必須很高，才有效果。
    #    將濾鏡數值矩陣的每個數值除以其最大值。再將濾鏡點對點乘上影像的RGB 值。模擬青光眼患者視野狹窄的現象。
    gl_img = glaucoma(img)

    cv2.imshow('im', rg_blind_img)
    # cv2.imwrite("hw2/rg_blind_img.jpg", rg_blind_img)
    # cv2.imwrite("hw2/yb_blind_img.jpg", yb_blind_img)
    # cv2.imwrite("hw2/gl_img.jpg", gl_img)

    # cv2.imshow("a", img)
    # cv2.imshow("b", rg_blind_img)
    # cv2.imshow("c", yb_blind_img)
    # cv2.imshow("d", gl_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows