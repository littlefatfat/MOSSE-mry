import numpy as np
import cv2
import os
from mryfx import get_img_lists
from mryfx import gauss_response
from mryfx import pre_process

learning_rate = lr = 0.125

# 输入图片路径
img_path = 'datasets/surfer/'

# 所有图片名称的列表
frame_list = get_img_lists(img_path)

print(frame_list[0])

# 第一帧图像读取，转灰度
init_frame = cv2.imread(frame_list[0])
init_frame_gray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)

# 图片宽高
img_height,img_width = init_frame_gray.shape

# 第一帧GT 左上宽高
init_gt = cv2.selectROI('gt', init_frame, False, False)  # 元组()
init_gt = np.array(init_gt).astype(np.int64)  # 数组[]

# F1
init_frame_cut = init_frame_gray[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]  # 第一帧剪下来gt图

# 为什么要变成浮点呢？ 不然后边就溢出了 预处理那块就一直输出0000000?
init_frame_cut_float = init_frame_cut.astype(np.float32)

# 第一帧图像预处理 传入切好的灰度图进行预处理
f1 = pre_process(init_frame_cut_float)

F1 = np.fft.fft2(f1)
F1_conj = np.conjugate(F1)

# G
# gt在原图坐标中心
init_center_x,init_center_y = init_gt[0]+0.5*init_gt[2],init_gt[1]+0.5*init_gt[3]
# 第一帧的高斯响应图
init_gauss_response = gauss_response(img_width, img_height, init_center_x, init_center_y)
# 从大高斯剪出来gt的高斯
gt_gauss = init_gauss_response[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
# 图像域 转 频域
G = np.fft.fft2(gt_gauss)

# A1  B1  H1*开始计算
A1 = G * F1_conj
B1 = F1 * F1_conj

H1_conj = A1/B1

# ！！！仿射变换！！！ 待会再写

# 接下来开始跟踪了吗终于要
for idx, image in enumerate(frame_list):

    current_image = cv2.imread(image)

    if idx == 0:

        box = init_gt
        A = A1
        B = B1
        H_conj = H1_conj
    else:
        image = current_image.astype(np.float32)  # 变成浮点数，不然后边预处理就溢出了
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转灰度

        f = image_gray[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]  # 切下来小块

        f = pre_process(f)
        F = np.fft.fft2(f)
        Gi = F * H_conj
        gi = np.fft.ifft2(Gi)

        top_value = np.max(gi)
        top = np.where(gi == top_value)
        # 偏移
        dx = int(np.mean(top[1])-box[2]/2)   # top 有很多点
        dy = int(np.mean(top[0])-box[3]/2)
        # 更新位置
        box[0] = box[0]+dx
        box[1] = box[1]+dy

        # 更新H
        f = image_gray[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]  # fi产生
        F = np.fft.fft2(f)
        F_conj = np.conjugate(F)
        A = lr * G * F_conj + (1-lr) * A
        B = lr * F * F_conj + (1-lr) * B
        H_conj = A / B

    # 显示 画框 左上右下 蓝色 粗度为2
    cv2.rectangle(current_image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255, 0, 0), 2)
    cv2.imshow('track', current_image)
    cv2.waitKey(100)

