import numpy as np
import cv2
import os

def generate_depth_map(data_root, file_id):
    # 1. 加载左右目图像
    img_left_path = os.path.join(data_root, "image_2", f"{file_id}.png")
    img_right_path = os.path.join(data_root, "image_3", f"{file_id}.png")
    calib_path = os.path.join(data_root, "calib", f"{file_id}.txt")

    img_l = cv2.imread(img_left_path, cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(img_right_path, cv2.IMREAD_GRAYSCALE)

    if img_l is None or img_r is None:
        print(f"跳过 {file_id}: 找不到图像")
        return

    # 2. 从校准文件获取基线(Baseline)和焦距(f)
    # KITTI 中: Z = (f * b) / disparity
    with open(calib_path, 'r') as f:
        lines = f.readlines()
        P2 = np.array([float(x) for x in lines[2].split()[1:]]).reshape(3, 4)
        P3 = np.array([float(x) for x in lines[3].split()[1:]]).reshape(3, 4)

    focal_length = P2[0, 0]
    # 基线 b = (P3_x - P2_x) / f (单位: 米)
    baseline = (P2[0, 3] - P3[0, 3]) / focal_length

    # 3. 配置 SGBM 立体匹配器
    # 这些参数对深度图的质量至关重要
    window_size = 5
    min_disp = 0
    num_disp = 16 * 5  # 必须是16的倍数，决定了能检测的最大深度

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    # 4. 计算视差图 (Disparity Map)
    disparity = stereo.compute(img_l, img_r).astype(np.float32) / 16.0

    # 5. 计算深度图: Z = (f * b) / d
    # 避免除以 0，将极小视差设为较大深度的有效值
    disparity[disparity <= 0] = 0.1
    depth_map = (focal_length * baseline) / disparity

    # 6. 可视化
    # 视差图可视化
    disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    # 深度图可视化（截断 80 米以内的范围，效果更好）
    depth_vis = np.clip(depth_map, 0, 80)
    depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_vis = 255 - depth_vis # 越近越亮
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

    cv2.imshow("Left Image", cv2.imread(img_left_path))
    cv2.imshow("Disparity (Stereo Matching)", disp_color)
    cv2.imshow("Depth Map (Reconstructed)", depth_color)

    print(f"ID: {file_id}, Baseline: {baseline:.4f}m, Focal: {focal_length:.2f}")
    cv2.waitKey(0)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(current_dir, "data", "object", "training")
    generate_depth_map(root, "000000")