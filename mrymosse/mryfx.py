import numpy as np
import cv2
import os

gause_sigma_square = 100

# 从图片路径文件夹读取所有图片，将图片名放在一个列表里
def get_img_lists(img_path):
    frame_list = []
    for frame in os.listdir(img_path):
        if frame[-4:] == '.jpg':
            frame_list.append(img_path+frame)
    frame_list.sort()  # 按顺序排
    return frame_list

# 0-1 标准化
def normalization_0_1(x):
    x_min = x.min()
    x_max = x.max()
    x = (x-x_min)/(x_max-x_min)
    return x

# 得到图片高斯响应
def gauss_response(width,height,center_x,center_y):

    # 图片中每个点的x,y坐标,相对于左上角点
    x_location = np.arange(width)  # [0,1,2,....,w-1]
    y_location = np.arange(height)  # [0,1,2,....,y-1]

    xx, yy = np.meshgrid(x_location, y_location)  # [[0,1,,,479],   [[0,0,,,0],
                                                  #  [0,1,,,479],    [1,1,,,1],
                                                  #
                                                  #  [0,1,,,479]]    [359,359,,,359]]
    # 高斯函数
    distance = (np.square(xx-center_x)+np.square(yy-center_y))/(2*gause_sigma_square)
    gause = np.exp(-distance)
    # 0-1标准化
    gause_response01 = normalization_0_1(gause)

    return gause_response01


# 二维余弦窗函数
def hanning_2d(width,height):
    wid_cos = np.hanning(width)
    hei_cos = np.hanning(height)
    xx,yy = np.meshgrid(wid_cos,hei_cos)
    windows = xx*yy
    return windows

# 图片预处理
def pre_process(image):

    image1 = np.log(image+1)   # log
    image2 = (image1 - np.mean(image1))/(np.std(image1)+1e-5)  # z-score标准化

    height,width = image2.shape

    image3 = image2 * hanning_2d(width,height)   # 余弦窗处理

    return image3

# image = cv2.imread('datasets/surfer/0001.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = image.astype(np.float32)
# image = pre_process(image)
# cv2.imshow('track', image)
# cv2.waitKey(10000)