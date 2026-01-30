import numpy as np
import cv2
class StableDepthEstimator:
    def __init__(self, max_depth=100.0, smoothing_factor=0.7):
        self.max_depth = max_depth
        self.smoothing_factor = smoothing_factor
        self.previous_depth_map = None

    def estimate_depth(self, img_l, img_r, P2, P3):
        focal_length = P2[0, 0]
        baseline = abs(P2[0, 3] - P3[0, 3]) / focal_length

        # 图像预处理
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_l_enhanced = clahe.apply(img_l)
        img_r_enhanced = clahe.apply(img_r)

        # 立体匹配
        methods = [
            ('SGBM', cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=16*8,
                blockSize=5,
                P1=8 * 3 * 5**2,
                P2=32 * 3 * 5**2,
                disp12MaxDiff=1,
                uniquenessRatio=15,
                speckleWindowSize=100,
                speckleRange=32
            )),
            ('BM', cv2.StereoBM_create(
                numDisparities=16*6,
                blockSize=15
            ))
        ]

        disparities = []
        for name, matcher in methods:
            if name == 'BM':
                disp = matcher.compute(img_l, img_r).astype(np.float32) / 16.0
            else:
                disp = matcher.compute(img_l_enhanced, img_r_enhanced).astype(np.float32) / 16.0
            disparities.append(disp)

        final_disparity = np.mean(disparities, axis=0)
        final_disparity[final_disparity <= 0] = 0.1

        depth_map = (focal_length * baseline) / final_disparity
        depth_map = cv2.medianBlur(depth_map, 5)

        # 时间域平滑
        if self.previous_depth_map is not None:
            depth_map = (self.smoothing_factor * depth_map +
                        (1 - self.smoothing_factor) * self.previous_depth_map)

        self.previous_depth_map = depth_map.copy()

        return depth_map

    def create_depth_visualization(self, depth_map):
        """创建稳定的深度可视化"""
        depth_normalized = np.clip(depth_map, 0, self.max_depth)
        depth_normalized = (depth_normalized / self.max_depth) * 255
        depth_normalized = depth_normalized.astype(np.uint8)
        return cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)