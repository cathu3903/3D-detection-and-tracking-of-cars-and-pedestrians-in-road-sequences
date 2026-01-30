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
    num_disparities = 16 * 10  # 增加搜索范围，KITTI最大视差可达256
    min_disparity = 0          # KITTI最小视差为0
    block_size = 5             # 适中大小，平衡精度和速度

    # 针对KITTI场景的道路和车辆特点优化参数
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,      # 匹配成本惩罚项1（较小梯度）
        P2=32 * 3 * block_size**2,     # 匹配成本惩罚项2（较大梯度）
        disp12MaxDiff=1,               # 左右一致性检查最大差异
        preFilterCap=63,               # 预过滤器截断值
        uniquenessRatio=15,            # 唯一性比率
        speckleWindowSize=150,         # 斑点滤波窗口大小
        speckleRange=2,               # 斑点滤波范围
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # 使用3路SGBM提高精度
    )

    # 预处理：KITTI图像增强
    # 应用CLAHE增强对比度，有助于在KITTI的光照变化中保持特征
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_l_enhanced = clahe.apply(img_l)
    img_r_enhanced = clahe.apply(img_r)

    # 计算视差
    disparity = stereo.compute(img_l_enhanced, img_r_enhanced).astype(np.float32) / 16.0

    # 处理无效区域
    # 基于KITTI数据集特点：远处物体视差小，近处物体视差大
    disparity[disparity <= 0] = 0.1  # 避免除零错误
    # 设置合理的最大视差上限
    max_valid_disparity = num_disparities - 1
    disparity[disparity > max_valid_disparity] = max_valid_disparity

    # 转换为深度图
    depth_map = (focal_length * baseline) / disparity

    # 针对KITTI场景的后处理
    # 应用双边滤波保持边缘，减少噪声
    depth_map = cv2.bilateralFilter(depth_map, 5, 100, 100)
    # 中值滤波去除椒盐噪声
    depth_map = cv2.medianBlur(depth_map, 5)

    # KITTI数据集深度裁剪（典型范围）
    # KITTI中大部分有效深度在3-80米范围内
    depth_map = np.clip(depth_map, 1.0, 100.0)  # 限制深度范围

    return depth_map

def get_depth_map_kitti_advanced(img_l, img_r, P2, P3):
    """
    更高级的KITTI优化版本，包含WLS滤波
    """
    focal_length = P2[0, 0]
    baseline = abs(P2[0, 3] - P3[0, 3]) / focal_length

    # 高质量参数设置
    num_disparities = 16 * 12  # 增加搜索范围
    block_size = 5

    # 左匹配器
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        preFilterCap=63,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_HH4  # 使用Hartley-Hong模式提高精度
    )

    # 右匹配器与WLS滤波器（需要opencv-contrib-python的ximgproc）
    right_matcher = None
    wls_filter = None
    try:
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
        wls_filter.setLambda(80000)  # 控制平滑程度
        wls_filter.setSigmaColor(1.8)  # 颜色相似性参数
    except AttributeError:
        # 如果没有opencv-contrib-python，跳过右匹配与WLS滤波
        print("ximgproc not available, skipping right matcher and WLS filter...")

    # 图像预处理
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_l_enhanced = clahe.apply(img_l)
    img_r_enhanced = clahe.apply(img_r)

    # 计算左右视差图
    disp_left = left_matcher.compute(img_l_enhanced, img_r_enhanced).astype(np.float32) / 16.0
    disp_right = None
    if right_matcher is not None:
        disp_right = right_matcher.compute(img_r_enhanced, img_l_enhanced).astype(np.float32) / 16.0

    if wls_filter is not None and disp_right is not None:
        # 应用WLS滤波
        filtered_disparity = wls_filter.filter(disp_left, img_l_enhanced, None, disp_right)
        # 处理无效区域
        filtered_disparity[filtered_disparity <= 0] = 0.1
    else:
        # 如果没有WLS，直接使用左视差图并进行后处理
        filtered_disparity = disp_left
        filtered_disparity[filtered_disparity <= 0] = 0.1
        filtered_disparity = np.clip(filtered_disparity, 0.1, num_disparities)

    # 转换为深度图
    depth_map = (focal_length * baseline) / filtered_disparity

    # 后处理
    depth_map = cv2.bilateralFilter(depth_map, 5, 100, 100)
    depth_map = cv2.medianBlur(depth_map, 5)
    depth_map = np.clip(depth_map, 1.0, 100.0)

    return depth_map

def get_depth_map(img_l, img_r, P2, P3):
    """
    默认使用优化版本
    """
    return get_depth_map_kitti_optimized(img_l, img_r, P2, P3)
