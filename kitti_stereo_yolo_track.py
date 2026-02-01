import cv2
import numpy as np
import os
from ultralytics import YOLO
#from stereo_depth_sgbc import StableDepthEstimator  # Stereo depth logic in stereo_depth_sgbc.py
from stereo_depth_r import get_depth_map_kitti_optimized


class KittiStereoYoloSystemV26:
    def __init__(self, drive_path, model_path='yolo26n.pt',iou=0.6, conf=0.3):
        self.drive_path = drive_path
        # Color images
        self.left_dir = os.path.join(drive_path, 'image_02/data')
        self.right_dir = os.path.join(drive_path, 'image_03/data')
        # Gray images
        # self.left_dir = os.path.join(drive_path, 'image_00/data')
        # self.right_dir = os.path.join(drive_path, 'image_01/data')
        self.img_list = sorted(os.listdir(self.left_dir))

        # 1) Detector: YOLOv26 model with built-in tracking
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

        # Only keep vehicles and pedestrians
        self.target_names = {'person', 'car', 'bus', 'truck', 'motorcycle', 'bicycle'}
        names = self.model.names
        if isinstance(names, dict):
            self.target_ids = {i for i, n in names.items() if n in self.target_names}
        else:
            self.target_ids = {i for i, n in enumerate(names) if n in self.target_names}

        # Initialize frame index
        self.frame_idx = 0


    def parse_raw_calib(self):
        calib_path = os.path.join(
            os.path.dirname(__file__),
            'data',
            'raw_data',
            '2011_09_26_calib',
            '2011_09_26',
            'calib_cam_to_cam.txt',
        )

        if not os.path.exists(calib_path):
            raise FileNotFoundError(f'Calibration file not found: {calib_path}')

        data = {}
        with open(calib_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                key, value = line.split(':', 1)
                data[key.strip()] = value.strip()

        def _parse_matrix(key, shape):
            if key not in data:
                raise KeyError(f'Missing {key} in calibration file')
            vals = [float(x) for x in data[key].split()]
            return np.array(vals, dtype=np.float32).reshape(shape)

        P2 = _parse_matrix('P_rect_02', (3, 4))
        P3 = _parse_matrix('P_rect_03', (3, 4))

        return P2, P3

    def _detect_and_track_with_yolo26(self, frame_bgr):
        # Using Ultralytics built-in tracking
        results = self.model.track(
            frame_bgr,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            tracker="botsort.yaml"  # Use ByteTrack as default, can also use botsort
        )

        if not results or len(results) == 0:
            return [], []

        boxes = []
        track_ids = []

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in self.target_ids:
                        continue

                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    track_id = int(box.id[0]) if box.id is not None else -1

                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    if w > 0 and h > 0:
                        boxes.append((x, y, w, h))
                        track_ids.append(track_id)

        return boxes, track_ids

    def run(self):
        P2, P3 = self.parse_raw_calib()

        # Dictionary to store colors for each track ID
        track_colors = {}
        #estimator = StableDepthEstimator()

        for img_name in self.img_list:
            frame_l = cv2.imread(os.path.join(self.left_dir, img_name))
            frame_r = cv2.imread(os.path.join(self.right_dir, img_name))
            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

            self.frame_idx += 1

            # Step 1: Detect and track with YOLOv26
            detections, track_ids = self._detect_and_track_with_yolo26(frame_l)

            # Step 2: Get depth map for 3D localization
            # sgbc
            # depth_map = estimator.estimate_depth(gray_l, gray_r, P2, P3)
            # depth_vis = estimator.create_depth_visualization(depth_map)
            # sgbm optimized
            depth_map = get_depth_map_kitti_optimized(gray_l, gray_r, P2, P3)
            # sgbm with WLS filter
            # depth_map = get_depth_map_kitti_advanced(gray_l, gray_r, P2, P3)

            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            cv2.imshow('Depth Map', depth_vis)

            # Step 3: Process detections and visualize
            for i, ((x, y, w, h), track_id) in enumerate(zip(detections, track_ids)):
                p1 = (x, y)
                p2 = (x + w, y + h)

                # Get or generate color based on track ID
                if track_id not in track_colors:
                    # Generate a consistent color for this track ID
                    # Using hash of track_id to ensure same ID always gets same color
                    np.random.seed(track_id % (2**32-1))  # Use modulo to stay within seed range
                    color = tuple(map(int, np.random.randint(0, 255, 3)))
                    track_colors[track_id] = color
                else:
                    color = track_colors[track_id]

                cv2.rectangle(frame_l, p1, p2, color, 2)

                # Display track ID
                track_text = f'Track:{track_id}' if track_id != -1 else f'Det:{i}'
                cv2.putText(frame_l, track_text, (p1[0], p1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Calculate depth from ROI
                z_roi = depth_map[y:y + h, x:x + w]
                if z_roi.size == 0:
                    continue
                z_val = float(np.median(z_roi))
                if not np.isfinite(z_val):
                    continue

                # Display depth information
                cv2.putText(frame_l, f'Z={z_val:.2f}m', (x, y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow('KITTI Tracking and 3D Localization with YOLOv26', frame_l)

            if (cv2.waitKey(30) % 256) == 27:  # ESC key to exit
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    drive_root = './data/raw_data/2011_09_26/2011_09_26_drive_0013_sync'
    system = KittiStereoYoloSystemV26(drive_root, model_path='yolo26n.pt')
    system.run()