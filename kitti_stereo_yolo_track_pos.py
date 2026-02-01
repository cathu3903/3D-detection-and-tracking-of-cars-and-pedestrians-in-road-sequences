import cv2
import numpy as np
import os
from ultralytics import YOLO
from stereo_depth_r import get_depth_map_kitti_optimized


class KittiStereoYoloSystemV26:
    def __init__(self, drive_path, model_path="yolo26n.pt", conf=0.4):
        self.drive_path = drive_path
        # Color images
        self.left_dir = os.path.join(drive_path, "image_02/data")
        self.right_dir = os.path.join(drive_path, "image_03/data")
        # Gray images
        # self.left_dir = os.path.join(drive_path, "image_00/data")
        # self.right_dir = os.path.join(drive_path, "image_01/data")
        self.img_list = sorted(os.listdir(self.left_dir))

        # 1) Detector: YOLOv26 model with built-in tracking
        self.model = YOLO(model_path)
        self.conf = conf

        # Only keep vehicles and pedestrians
        self.target_names = {"person", "car", "bus", "truck", "motorcycle", "bicycle"}
        names = self.model.names
        if isinstance(names, dict):
            self.target_ids = {i for i, n in names.items() if n in self.target_names}
        else:
            self.target_ids = {i for i, n in enumerate(names) if n in self.target_names}

        # Initialize frame index
        self.frame_idx = 0

    def _read_kitti_calib_file(self, calib_path):
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
        return data

    def parse_raw_calib(self):
        calib_root = os.path.join(
            os.path.dirname(__file__),
            "data",
            "raw_data",
            "2011_09_26_calib",
            "2011_09_26",
        )

        cam_to_cam_path = os.path.join(calib_root, "calib_cam_to_cam.txt")
        velo_to_cam_path = os.path.join(calib_root, "calib_velo_to_cam.txt")

        cam_data = self._read_kitti_calib_file(cam_to_cam_path)
        velo_data = self._read_kitti_calib_file(velo_to_cam_path)

        def _parse_matrix(data, key, shape):
            if key not in data:
                raise KeyError(f"Missing {key} in calibration file")
            vals = [float(x) for x in data[key].split()]
            return np.array(vals, dtype=np.float32).reshape(shape)

        P2 = _parse_matrix(cam_data, "P_rect_02", (3, 4))
        P3 = _parse_matrix(cam_data, "P_rect_03", (3, 4))
        # Use rectification for cam2
        R_rect = _parse_matrix(cam_data, "R_rect_02", (3, 3))

        # calib_velo_to_cam.txt provides R and T
        R_velo_to_cam = _parse_matrix(velo_data, "R", (3, 3))
        T_velo_to_cam = _parse_matrix(velo_data, "T", (3, 1))

        # Build 4x4 transforms
        R_rect_4x4 = np.eye(4, dtype=np.float32)
        R_rect_4x4[:3, :3] = R_rect

        Tr_v2c_4x4 = np.eye(4, dtype=np.float32)
        Tr_v2c_4x4[:3, :3] = R_velo_to_cam
        Tr_v2c_4x4[:3, 3:4] = T_velo_to_cam

        # rectified cam -> velodyne (vehicle) transform
        # Transform matrix from velodyne to rectified cam2 -> from cam2 to velodyne (vehicle)
        T_cam2_to_velo = np.linalg.inv(Tr_v2c_4x4) @ np.linalg.inv(R_rect_4x4)

        return P2, P3, T_cam2_to_velo

    def _detect_and_track_with_yolo26(self, frame_bgr):
        results = self.model.track(
            frame_bgr,
            persist=True,       # enable tracking
            conf=self.conf,     # confidence threshold set to 0.5, reduce the detection noise
            verbose=False,      # not print debug info
            tracker="bytetrack.yaml", # charge the default parameters of tracker botsort
        )

        if not results or len(results) == 0:
            return [], []

        boxes = []
        track_ids = []

        for r in results:
            # Skip if no boxes
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.target_ids:
                    continue
                # Get target box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                track_id = int(box.id[0]) if box.id is not None else -1

                # convert into cv2 box
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                if w > 0 and h > 0:
                    boxes.append((x, y, w, h))
                    track_ids.append(track_id)

        return boxes, track_ids

    def run(self):
        P2, P3, T_cam2_to_velo = self.parse_raw_calib()

        # Dictionary to store colors for each track ID
        track_colors = {}

        # Cache intrinsics
        fx, fy = P2[0, 0], P2[1, 1]
        cx, cy = P2[0, 2], P2[1, 2]

        for img_name in self.img_list:
            frame_l = cv2.imread(os.path.join(self.left_dir, img_name))
            frame_r = cv2.imread(os.path.join(self.right_dir, img_name))
            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

            self.frame_idx += 1

            # Step 1: Detect and track with YOLOv26
            detections, track_ids = self._detect_and_track_with_yolo26(frame_l)

            # Step 2: Get depth map for 3D localization
            depth_map = get_depth_map_kitti_optimized(gray_l, gray_r, P2, P3)

            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imshow("Depth Map", depth_vis)

            # Step 3: Calcluate the position in vehicule axis and visualize
            for i, ((x, y, w, h), track_id) in enumerate(zip(detections, track_ids)):
                p1 = (x, y)
                p2 = (x + w, y + h)

                if track_id not in track_colors:
                    np.random.seed(track_id % (2**32 - 1))
                    color = tuple(map(int, np.random.randint(0, 255, 3)))
                    track_colors[track_id] = color
                else:
                    color = track_colors[track_id]

                cv2.rectangle(frame_l, p1, p2, color, 2)

                track_text = f"Track:{track_id}" if track_id != -1 else f"Det:{i}"
                cv2.putText(
                    frame_l,
                    track_text,
                    (p1[0], p1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

                # Calculate depth from ROI
                z_roi = depth_map[y : y + h, x : x + w]
                if z_roi.size == 0:
                    continue
                z_val = float(np.median(z_roi))
                if not np.isfinite(z_val):
                    continue

                # Pixel center in left image
                u = x + 0.5 * w
                v = y + 0.5 * h

                # Camera (rectified cam2) coordinates
                Xc = (u - cx) * z_val / fx
                Yc = (v - cy) * z_val / fy
                Zc = z_val

                # Convert to vehicle coords
                cam_h = np.array([Xc, Yc, Zc, 1.0], dtype=np.float32)
                velo_h = T_cam2_to_velo @ cam_h
                Xv, Yv, Zv = velo_h[:3]

                # Depth text
                cv2.putText(
                    frame_l,
                    f"Z={z_val:.2f}m",
                    (x, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )
                # KITTI Velodyne(vehicle) coords: X-forward, Y-left, Z-up
                cv2.putText(
                    frame_l,
                    f"Velo(F,L,U)=({Xv:.2f},{Yv:.2f},{Zv:.2f})",
                    (x, y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

            cv2.imshow("KITTI Tracking and 3D Localization (Vehicle Coords)", frame_l)

            if (cv2.waitKey(30) % 256) == 27:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    drive_root = "./data/raw_data/2011_09_26/2011_09_26_drive_0013_sync"
    system = KittiStereoYoloSystemV26(drive_root, model_path="yolo26n.pt")
    system.run()
