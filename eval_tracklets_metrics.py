import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict

from kitti_stereo_yolo_track_pos import KittiStereoYoloSystemV26
from stereo_depth_r import get_depth_map_kitti_optimized

def parse_tracklets_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    tracklets = []
    # Only iterate top-level tracklets, not pose items
    for item in root.findall(".//tracklets/item"):
        obj_type = item.findtext("objectType")
        if obj_type is None:
            continue
        h_text = item.findtext("h")
        w_text = item.findtext("w")
        l_text = item.findtext("l")
        first_text = item.findtext("first_frame")
        if h_text is None or w_text is None or l_text is None or first_text is None:
            continue
        h = float(h_text)
        w = float(w_text)
        l = float(l_text)
        first_frame = int(first_text)

        poses = []
        poses_node = item.find("poses")
        if poses_node is None:
            continue
        for pose in poses_node.findall("item"):
            tx = float(pose.findtext("tx"))
            ty = float(pose.findtext("ty"))
            tz = float(pose.findtext("tz"))
            rx = float(pose.findtext("rx"))
            ry = float(pose.findtext("ry"))
            rz = float(pose.findtext("rz"))
            state = int(pose.findtext("state"))
            occ_text = pose.findtext("occlusion")
            trunc_text = pose.findtext("truncation")
            occlusion = int(occ_text) if occ_text is not None else -1
            truncation = int(trunc_text) if trunc_text is not None else -1
            poses.append((tx, ty, tz, rx, ry, rz, state, occlusion, truncation))

        tracklets.append({
            "type": obj_type,
            "dims": (l, w, h),
            "first_frame": first_frame,
            "poses": poses,
        })
    return tracklets

def rotation_matrix_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float32)

def build_3d_box(l, w, h):
    # Velodyne coords: x forward, y left, z up
    # Tracklets use object location at ground contact (bottom center).
    x = l / 2.0
    y = w / 2.0
    corners = np.array([
        [ x,  y,  h],
        [ x, -y,  h],
        [-x, -y,  h],
        [-x,  y,  h],
        [ x,  y,  0.0],
        [ x, -y,  0.0],
        [-x, -y,  0.0],
        [-x,  y,  0.0],
    ], dtype=np.float32)
    return corners

def project_box_to_image(corners_velo, T_velo_to_cam2, P2):
    # corners_velo: (8,3)
    ones = np.ones((8, 1), dtype=np.float32)
    pts_h = np.hstack([corners_velo, ones])  # (8,4)
    pts_cam = (T_velo_to_cam2 @ pts_h.T).T  # (8,4)
    pts_cam = pts_cam[:, :3]
    z = pts_cam[:, 2].copy()

    # project to image
    pts_cam_h = np.hstack([pts_cam, np.ones((8,1), dtype=np.float32)])
    pts_img_h = (P2 @ pts_cam_h.T).T  # (8,3)
    pts_img = pts_img_h[:, :2] / pts_img_h[:, 2:3]
    return pts_img, z

def bbox_from_points(pts):
    x1 = float(np.min(pts[:,0]))
    y1 = float(np.min(pts[:,1]))
    x2 = float(np.max(pts[:,0]))
    y2 = float(np.max(pts[:,1]))
    return (x1, y1, x2, y2)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def greedy_match(preds, gts, iou_th=0.5):
    matches = []
    used_p = set()
    used_g = set()
    for gi, g in enumerate(gts):
        best_iou, best_pi = 0.0, -1
        for pi, p in enumerate(preds):
            if pi in used_p:
                continue
            i = iou_xyxy(p["bbox"], g["bbox"])
            if i > best_iou:
                best_iou, best_pi = i, pi
        if best_iou >= iou_th:
            used_p.add(best_pi)
            used_g.add(gi)
            matches.append((best_pi, gi))
    return matches, used_p, used_g

def evaluate(drive_root, tracklet_xml, model_path="yolo26n.pt", visualize=True, vis_stride=10):
    system = KittiStereoYoloSystemV26(drive_root, model_path=model_path)

    P2, P3, T_cam2_to_velo = system.parse_raw_calib()
    T_velo_to_cam2 = np.linalg.inv(T_cam2_to_velo)

    tracklets = parse_tracklets_xml(tracklet_xml)

    # build GT by frame
    gt_by_frame = defaultdict(list)
    for gt_id, tr in enumerate(tracklets):
        l, w, h = tr["dims"]
        first = tr["first_frame"]
        for i, (tx, ty, tz, rx, ry, rz, state, occlusion, truncation) in enumerate(tr["poses"]):
            if state == 0:
                continue
            if occlusion > 0 or truncation > 0:
                continue
            frame = first + i

            # Build 3D box in velodyne coords
            corners = build_3d_box(l, w, h)
            R = rotation_matrix_z(rz)  # assumption: rz is yaw around Z in Velodyne
            corners = (R @ corners.T).T
            corners += np.array([tx, ty, tz], dtype=np.float32)

            pts_img, z = project_box_to_image(corners, T_velo_to_cam2, P2)
            if np.any(z <= 0):
                continue
            bbox = bbox_from_points(pts_img)

            # Z in cam coords for MAE
            center_velo = np.array([tx, ty, tz, 1.0], dtype=np.float32)
            center_cam = (T_velo_to_cam2 @ center_velo)[:3]
            z_cam = float(center_cam[2])

            gt_by_frame[frame].append({
                "track_id": gt_id,
                "bbox": bbox,
                "z": z_cam,
            })

    tp = 0
    fp = 0
    mae_sum = 0.0
    mae_count = 0
    id_switches = 0
    last_match_for_gt = {}

    for frame_idx, img_name in enumerate(system.img_list):
        frame_l = cv2.imread(os.path.join(system.left_dir, img_name))
        frame_r = cv2.imread(os.path.join(system.right_dir, img_name))
        if frame_l is None or frame_r is None:
            continue

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        detections, track_ids = system._detect_and_track_with_yolo26(frame_l)
        depth_map = get_depth_map_kitti_optimized(gray_l, gray_r, P2, P3)

        preds = []
        for (x, y, w, h), tid in zip(detections, track_ids):
            z_roi = depth_map[y : y + h, x : x + w]
            if z_roi.size == 0:
                continue
            z_val = float(np.median(z_roi))
            if not np.isfinite(z_val):
                continue
            preds.append({
                "track_id": tid,
                "bbox": (x, y, x + w, y + h),
                "z": z_val,
            })

        gts = gt_by_frame.get(frame_idx, [])
        h_img, w_img = frame_l.shape[:2]
        gts = [
            g for g in gts
            if not (g["bbox"][2] < 0 or g["bbox"][3] < 0 or g["bbox"][0] > w_img - 1 or g["bbox"][1] > h_img - 1)
        ]
        matches, used_p, used_g = greedy_match(preds, gts, iou_th=0.5)

        if visualize and (frame_idx % vis_stride == 0):
            vis = frame_l.copy()
            for g in gts:
                x1, y1, x2, y2 = map(int, g["bbox"])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"GT:{g['track_id']}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            for p in preds:
                x1, y1, x2, y2 = map(int, p["bbox"])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(vis, f"Pred:{p['track_id']}", (x1, y2 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("GT (green) vs Pred (red)", vis)
            if (cv2.waitKey(1) % 256) == 27:
                break

        tp += len(matches)
        fp += max(0, len(preds) - len(matches))

        for pi, gi in matches:
            p = preds[pi]
            g = gts[gi]
            mae_sum += abs(p["z"] - g["z"])
            mae_count += 1

            gt_id = g["track_id"]
            if gt_id in last_match_for_gt and last_match_for_gt[gt_id] != p["track_id"]:
                id_switches += 1
            last_match_for_gt[gt_id] = p["track_id"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    mae = mae_sum / mae_count if mae_count > 0 else float("inf")

    print(f"Precision: {precision:.4f}")
    print(f"MAE (Z):   {mae:.4f} m")
    print(f"ID switches: {id_switches}")

if __name__ == "__main__":
    drive_root = "./data/raw_data/2011_09_26/2011_09_26_drive_0048_sync"
    tracklet_xml = "./data/raw_data/2011_09_26/2011_09_26_drive_0048_sync/tracklet_labels.xml"
    evaluate(drive_root, tracklet_xml, model_path="yolo26n.pt")
