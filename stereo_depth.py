import numpy as np
import cv2

def get_depth_map(img_l, img_r, P2, P3):
    """
    输入左右图（灰度）和投影矩阵，返回深度图
    """
    # 计算焦距和基线
    focal_length = P2[0, 0]
    baseline = (P2[0, 3] - P3[0, 3]) / focal_length

    # SGBM 参数配置
    window_size = 5
    num_disp = 16 * 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    # 计算视差并转为深度
    disparity = stereo.compute(img_l, img_r).astype(np.float32) / 16.0
    disparity[disparity <= 0] = 0.1 # 避免除零
    depth_map = (focal_length * baseline) / disparity

    return depth_map