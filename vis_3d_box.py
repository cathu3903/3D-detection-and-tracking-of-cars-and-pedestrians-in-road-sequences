import numpy as np
import cv2
import os

class KittiVisualizer:
    def __init__(self, data_root):
        self.data_root = data_root
        # 预先获取并排序所有图片 ID
        self.image_dir = os.path.join(data_root, "image_2")
        if os.path.exists(self.image_dir):
            self.file_ids = sorted([f.split('.')[0] for f in os.listdir(self.image_dir) if f.endswith('.png')])
        else:
            self.file_ids = []

    def read_calib(self, file_id):
        calib_path = os.path.join(self.data_root, "calib", f"{file_id}.txt")
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line.startswith('P2:'):
                    p2_raw = np.array([float(x) for x in line.split()[1:]])
                    return p2_raw.reshape(3, 4)
        return None

    def load_labels(self, file_id):
        label_path = os.path.join(self.data_root, "label_2", f"{file_id}.txt")
        objects = []
        if not os.path.exists(label_path):
            return objects
        with open(label_path, 'r') as f:
            for line in f.readlines():
                data = line.split()
                if data[0] == 'DontCare': continue
                obj = {
                    'type': data[0],
                    'h': float(data[8]), 'w': float(data[9]), 'l': float(data[10]),
                    'pos': (float(data[11]), float(data[12]), float(data[13])),
                    'ry': float(data[14])
                }
                objects.append(obj)
        return objects

    def get_3d_box_corners(self, obj):
        h, w, l = obj['h'], obj['w'], obj['l']
        x, y, z = obj['pos']
        ry = obj['ry']
        R = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] += x
        corners_3d[1, :] += y
        corners_3d[2, :] += z
        return corners_3d

    def draw_projected_box(self, image, corners_3d, p2):
        corners_3d_homo = np.vstack([corners_3d, np.ones((1, 8))])
        projected = np.dot(p2, corners_3d_homo)
        projected[:2, :] /= projected[2, :]
        p2d = projected[:2, :].T.astype(np.int32)
        lines = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
        for s, e in lines:
            cv2.line(image, tuple(p2d[s]), tuple(p2d[e]), (0, 255, 0), 2)
        # 标记正面
        cv2.line(image, tuple(p2d[0]), tuple(p2d[5]), (0, 0, 255), 1)
        cv2.line(image, tuple(p2d[1]), tuple(p2d[4]), (0, 0, 255), 1)

    def run_browser(self):
        if not self.file_ids:
            print("未发现图片数据，请检查路径。")
            return

        idx = 0
        print("--- KITTI 浏览器已启动 ---")
        print("按键说明: [D] 下一张 | [A] 上一张 | [ESC] 退出")

        while True:
            file_id = self.file_ids[idx]
            img = cv2.imread(os.path.join(self.image_dir, f"{file_id}.png"))
            p2 = self.read_calib(file_id)
            objects = self.load_labels(file_id)

            for obj in objects:
                corners_3d = self.get_3d_box_corners(obj)
                self.draw_projected_box(img, corners_3d, p2)
                cv2.putText(img, f"{obj['type']} Z:{obj['pos'][2]:.2f}m",
                            (int(corners_3d[0,0]), int(corners_3d[1,0]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow("KITTI 3D Browser", img)
            key = cv2.waitKey(0) & 0xFF

            if key == 27: # ESC 退出
                break
            elif key == ord('d'): # 下一张
                idx = (idx + 1) % len(self.file_ids)
            elif key == ord('a'): # 上一张
                idx = (idx - 1) % len(self.file_ids)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(current_dir, "data", "object", "training")

    vis = KittiVisualizer(root)
    vis.run_browser()