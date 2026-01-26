import cv2
import numpy as np
import os
from stereo_depth import get_depth_map # 假设之前写的深度逻辑在stereo_depth.py中

class KittiStereoSystem:
    def __init__(self, drive_path):
        self.drive_path = drive_path
        self.left_dir = os.path.join(drive_path, "image_02/data")
        self.right_dir = os.path.join(drive_path, "image_03/data")
        self.img_list = sorted(os.listdir(self.left_dir))

        # 1. 加载检测器 (AdaBoost / Haar Cascade) [cite: 21]
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

        # 2. 初始化跟踪器 (使用简单的多目标跟踪逻辑)
        self.trackers = cv2.legacy.MultiTracker_create()

    def _backproject(self, u, v, z, P2):
        fx, fy = P2[0, 0], P2[1, 1]
        cx, cy = P2[0, 2], P2[1, 2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float32)

    def _project_points(self, pts3d, P2):
        pts_h = np.hstack([pts3d, np.ones((pts3d.shape[0], 1), dtype=np.float32)])
        pts2d = (P2 @ pts_h.T).T
        pts2d = pts2d[:, :2] / pts2d[:, 2:3]
        return pts2d.astype(int)

    def _make_bbox_corners(self, center, dims):
        h, w, l = dims
        x, y, z = center
        x_off = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2])
        y_off = np.array([   0,    0,    0,    0,   -h,   -h,   -h,   -h])
        z_off = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2])
        corners = np.vstack([x + x_off, y + y_off, z + z_off]).T
        return corners

    def _draw_3d_box(self, frame, corners2d, color=(0, 255, 0), thickness=2):
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i, j in edges:
            cv2.line(frame, tuple(corners2d[i]), tuple(corners2d[j]), color, thickness)

    def parse_raw_calib(self):
        # Raw Data 的校准文件通常在 drive 根目录或外层
        # 这里简化处理，直接定义计算 3D 需用的 P2, P3 逻辑
        # 实际操作中应读取 calib_cam_to_cam.txt
        calib_path = os.path.join(
            os.path.dirname(__file__),
            "data",
            "raw_data",
            "2011_09_26_calib",
            "2011_09_26",
            "calib_cam_to_cam.txt",
        )

        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")

        data = {}
        with open(calib_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()

        def _parse_matrix(key, shape):
            if key not in data:
                raise KeyError(f"Missing {key} in calibration file")
            vals = [float(x) for x in data[key].split()]
            return np.array(vals, dtype=np.float32).reshape(shape)

        # KITTI raw calibration uses rectified projection matrices
        P2 = _parse_matrix("P_rect_02", (3, 4))
        P3 = _parse_matrix("P_rect_03", (3, 4))

        return P2, P3

    def run(self):
        P2, P3 = self.parse_raw_calib()
        dims = (1.5, 1.6, 3.9)  # h, w, l 车辆尺寸先验（米）

        for img_name in self.img_list:
            # 读取左右图
            frame_l = cv2.imread(os.path.join(self.left_dir, img_name))
            frame_r = cv2.imread(os.path.join(self.right_dir, img_name))
            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

            # --- 步骤 1: 检测 (每隔10帧重新检测或在初始帧检测) ---
            # 为了演示，此处简化为若跟踪器为空则执行检测 [cite: 7]
            if len(self.trackers.getObjects()) == 0:
                cars = self.detector.detectMultiScale(gray_l, 1.1, 3)
                for (x, y, w, h) in cars:
                    self.trackers.add(cv2.legacy.TrackerKCF_create(), frame_l, (x, y, w, h))

            # --- 步骤 2: 跟踪 (Tracking) [cite: 8] ---
            success, boxes = self.trackers.update(frame_l)

            # --- 步骤 3: 3D 定位 (Localization) [cite: 9, 25] ---
            # 生成深度图 (利用之前写的逻辑)
            depth_map = get_depth_map(gray_l, gray_r, P2, P3)

            # 可视化深度图（归一化）
            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imshow("Depth Map", depth_vis)

            for i, box in enumerate(boxes):
                x, y, w, h = map(int, box)
                p1 = (x, y)
                p2 = (x + w, y + h)
                cv2.rectangle(frame_l, p1, p2, (255, 0, 0), 2)
                cv2.putText(frame_l, f"ID:{i}", (p1[0], p1[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 在此处根据box区域从depth_map获取中值，实现3D定位
                z_roi = depth_map[y:y+h, x:x+w]
                if z_roi.size == 0:
                    continue
                z_val = float(np.median(z_roi))
                if not np.isfinite(z_val):
                    continue
                cv2.putText(frame_l, f"Z={z_val:.2f}m", (x, y-25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                u = x + w / 2.0
                v = y + h / 2.0
                center3d = self._backproject(u, v, z_val, P2)
                corners3d = self._make_bbox_corners(center3d, dims)
                corners2d = self._project_points(corners3d, P2)
                self._draw_3d_box(frame_l, corners2d)

            cv2.imshow("KITTI Tracking & 3D Localization", frame_l)
            if cv2.waitKey(30) & 0xFF == 27: break

        cv2.destroyAllWindows()

# 使用示例
drive_root = "./data/raw_data/2011_09_26/2011_09_26_drive_0005_sync"
system = KittiStereoSystem(drive_root)
system.run()
