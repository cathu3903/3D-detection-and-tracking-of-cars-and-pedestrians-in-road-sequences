import numpy as np
import cv2

def get_depth_map_kitti_optimized(img_l, img_r, P2, P3):
    """
    针对KITTI数据集优化的深度估计
    """
    # 计算焦距和基线
    focal_length = P2[0, 0]
    baseline = abs(P2[0, 3] - P3[0, 3]) / focal_length

    # KITTI特定优化参数
    # KITTI数据集的视差范围通常在0-256之间
    num_disparities = 16 * 14  # 增加搜索范围，KITTI最大视差可达256
    min_disparity = 0          # KITTI最小视差为0
    block_size = 7             # 适中大小，平衡精度和速度

    # 针对KITTI场景的道路和车辆特点优化参数
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,      # 匹配成本惩罚项1（较小梯度）
        P2=32 * 3 * block_size**2,     # 匹配成本惩罚项2（较大梯度）
        disp12MaxDiff=1,               # 左右一致性检查最大差异
        preFilterCap=63,               # 预过滤器截断值
        uniquenessRatio=10,            # 唯一性比率
        speckleWindowSize=200,         # 斑点滤波窗口大小
        speckleRange=4,               # 斑点滤波范围
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # 使用3路SGBM提高精度
    )

    # 计算视差
    disparity = stereo.compute(img_l, img_r).astype(np.float32) / 16.0

    # 处理无效区域
    # 基于KITTI数据集特点：远处物体视差小，近处物体视差大
    disparity[disparity <= 0] = 0.1  # 避免除零错误
    # 设置合理的最大视差上限
    max_valid_disparity = num_disparities - 1
    disparity[disparity > max_valid_disparity] = max_valid_disparity

    # 转换为深度图
    depth_map = (focal_length * baseline) / disparity

    depth_map[:, :num_disparities] = 0

    # KITTI数据集深度裁剪（典型范围）
    # KITTI中大部分有效深度在3-80米范围内
    depth_map = np.clip(depth_map, 1.0, 100.0)  # 限制深度范围

    return depth_map


def get_depth_map(img_l, img_r, P2, P3):
    """
    默认使用优化版本
    """
    return get_depth_map_kitti_optimized(img_l, img_r, P2, P3)
