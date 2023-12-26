import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

data_path = "C:/Users/cghsi/Desktop/HW3/train1000/"

# 第一層捲積
fil1 = np.array([-1, -1, 1, -1, 0, 1, -1, 1, 1]).reshape(3, 3)
# 第二層捲積
fil2 = fil1.T


def readImg(num1, num2):

    train = []

    num1 = num1 + 1 if num1 == 0 else num1 * 100
    num2 = num2 + 1 if num2 == 0 else num2 * 100

    for i in range(num1, num1 + 100):
        img = cv2.imread(data_path + str(i) + ".png", 0)
        img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
        train.append(img)

    for i in range(num2, num2 + 100):
        img = cv2.imread(data_path + str(i) + ".png", 0)
        img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
        train.append(img)
    return train


def maxPooling(feature_map, size=2, stripe=2):
    in_ax0, in_ax1 = np.shape(feature_map)
    # output size
    output_ax0 = int((in_ax0 - size) // stripe + 1)
    output_ax1 = int((in_ax1 - size) // stripe + 1)
    output = np.zeros((output_ax0, output_ax1))

    for row_ind in range(0, output_ax0):
        for col_inx in range(0, output_ax1):
            start_y = row_ind * stripe
            start_x = col_inx * stripe
            crop_map = feature_map[start_y:start_y + size, start_x:start_x + size]
            max_value = np.max(crop_map)
            output[row_ind, col_inx] = max_value

    return output


def layer1(input1, kernel):
    input_conv = cv2.filter2D(input1, -1, kernel)
    output = maxPooling(input_conv)
    return output


def layer2(input2, kernel):
    a = cv2.filter2D(input2[:, :, 0], -1, kernel) + cv2.filter2D(input2[:, :, 1], -1, kernel)
    output = maxPooling(input_conv)
    return output


def trainModel(num1, num2):
    # Read training data (return a list)
    train_dataset = readImg(num1, num2)

    # label X is the feature map after convolution
    X = []
    # label Y is the target number
    train_nums = len(train_dataset)
    Y = np.zeros(train_nums)
    Y[:int(train_nums/2)] += num1
    Y[int(train_nums/2):] += num2
    Y = np.array(Y).reshape(train_nums, 1)

    for img in train_dataset:
        # 1st convolution and max pooling
        img = np.array([layer1(img, fil1), layer1(img, fil2)]).transpose(1, 2, 0)
        # 2nd convolution and max pooling
        img = np.array([layer2(img, fil1), layer2(img, fil2)]).transpose(1, 2, 0)
        # flatten array and append a constant 1 in the end
        img = np.ndarray.flatten(img)
        img = np.append(img, 1)
        X.append(img)
    X = np.array(X)
    # Linear regression
    A = np.dot(np.linalg.inv(np.dot(X.T, X)), (np.dot(X.T, Y)))
    # Predict value
    Yp = np.dot(X, A)
    # print(Yp)
    # y_pred = np.where(Yp > 0.5, num2, num1)



if __name__ == "__main__":
    x = trainModel(0, 9)
