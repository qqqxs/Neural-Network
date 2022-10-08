# -*- coding = utf -8 -*-
# @Time : 2022/9/4 17:18
# @File : Dichotomies of cat pictures by using neural network.py
# process : 激活函数->初始化参数->正反向传播函数->梯度下降函数->预测函数
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 设置选项允许重复加载动态链接库。os.environ表示字符串环境的mapping对象


def load_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


"""每张图片的大小为64*64色素点，每个色素点由RGB三原色组成，所以每张图片的数据维度为（64，64，3），一张图片共需要12288个数据点确定。load_dataset函数的返回值意义如下：
train_set_x_orig:训练集图像数据，一共209张，数据维度为(209,64,64,3)
train_set_y_orig:训练集的标签集，维度为(1,209)
test_set_x_orig:测试集图像数据，一共50张，维度为(50,64,64,3)
test_set_y_orig:测试集的标签集，维度为(1,50)
classes ： 保存的是以bytes类型保存的两个字符串数据，数据为：[b’non-cat’ b’cat’]
"""
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

index = 5
plt.imshow(train_set_x_orig[index])
plt.show()
print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode(
    "utf-8") + "' picture.")

m_train = train_set_x_orig.shape[0]  # 训练集的照片数量
m_test = test_set_x_orig.shape[0]  # 测试集的照片数量
num_px = train_set_x_orig.shape[1]  # 训练集的照片高度（宽度）0

print("训练集图片数量：" + str(m_train))  # 打印基本图像信息
print("测试剂图片数量：" + str(m_test))
print("训练集和测试集图片的高度和宽度：" + str(num_px))
print("每张图片大小：(" + str(num_px) + ", " + str(num_px) + ", " + str(train_set_x_orig.shape[-1]) + ")")
print('测试集图片的维数为：' + str(train_set_x_orig.shape))  # (209,64,64,3)
print('测试集标签的维数为：' + str(train_set_y.shape))  # (1,209)
print('训练集图片的维数为：' + str(test_set_x_orig.shape))  # (50,64,64,3)
print('训练集标签的维数：' + str(test_set_y.shape))  # (1,50)

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T  # 将图片降维并转置，处理后的shape为（64*64*3，209）
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print('训练集图片降维后的维数为：' + str(train_set_x_flatten.shape))  # (12288,209)
print('训练集标签的维数为：' + str(train_set_y.shape))  # (1,209)
print('测试集图片降维后的维数为：' + str(test_set_x_flatten.shape))  # (12288,50)
print('测试集标签的维数为：' + str(test_set_y.shape))  # (1,50)

train_set_x = train_set_x_flatten / 255  # 标准化
test_set_x = test_set_x_flatten / 255


def sigmoid(z):  # 定义sigmoid函数
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_params(dim):  # 定义初始化训练参数的函数
    w = np.zeros((dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b


def propagate(w, b, X, Y):  # 定义向前，向后传播函数
    m = X.shape[1]
    A = sigmoid((np.dot(w.T, X) + b))  # 前向传播
    cost = (-1 / m) * np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A)))
    dw = (1 / m) * np.dot(X, (A - Y).T)  # 反向传播
    db = (1 / m) * np.sum(A - Y)
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = {"dw": dw,
             "db": db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate):  # 定义梯度下降函数
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        if i % 100 == 0:
            print('迭代%d次，误差为%f' % (i, cost))
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs


def predict(w, b, X):  # 定义预测函数
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(m):
        Y_prediction[0, i] = 1
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
    assert (Y_prediction.shape == (1, m))
    return Y_prediction


def model(train_x, train_y, test_x, test_y, num_iterations=2000, learning_rate=0.05):
    w, b = initialize_params(train_x.shape[0])
    params, grads, costs = optimize(w, b, train_x, train_y, num_iterations, learning_rate)
    w = params['w']
    b = params['b']
    Y_prediction_train = predict(w, b, train_x)
    Y_prediction_test = predict(w, b, test_x)
    print('训练集准确性 ： ', format((1 - np.mean(np.abs(Y_prediction_train - train_y))) * 100), '%')
    print('测试集准确性 ： ', format((1 - np.mean(np.abs(Y_prediction_test - test_y))) * 100), '%')
    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations}
    return d


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005)

img1 = "D:\pythonProject\Dichotomies of cat pictures by using nutral network\mjnz.jpg"
img2 = "D:\pythonProject\Dichotomies of cat pictures by using nutral network\zhjj.jpg"
image1 = np.array(plt.imread(img1))
image2 = np.array(plt.imread(img2))
my_image1 = np.array(Image.fromarray(image1).resize((num_px, num_px))).reshape((1, num_px * num_px * 3)).T
my_predicted_image1 = predict(d["w"], d["b"], my_image1)
my_image2 = np.array(Image.fromarray(image2).resize((num_px, num_px))).reshape((1, num_px * num_px * 3)).T
my_predicted_image2 = predict(d["w"], d["b"], my_image2)
plt.imshow(image1)
plt.show()
plt.imshow(image2)
plt.show()
print("y = " + str(np.squeeze(my_predicted_image1)) + ", your algorithm predicts a \"" + classes[
    int(np.squeeze(my_predicted_image1))].decode("utf-8") + "\" picture.")
print("y = " + str(np.squeeze(my_predicted_image2)) + ", your algorithm predicts a \"" + classes[
    int(np.squeeze(my_predicted_image2))].decode("utf-8") + "\" picture.")
