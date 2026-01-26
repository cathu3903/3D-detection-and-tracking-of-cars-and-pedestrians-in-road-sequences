import numpy as np
import cv2
import os
# 导入你的深度生成函数和之前的可视化工具类
from stereo_depth import get_depth_map

class KittiEvaluator:
    def __init__(self, data_root):
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, "image_2")
        self.file_ids = sorted([f.split('.')[0] for f in os.listdir(self.image_dir) if f.endswith('.png')])

    def parse_calib(self, file_id):
        calib_path = os.path.join(self.data_root, "calib", f"{file_id}.txt")
        with open(calib_path, 'r') as f:
            lines = f.readlines()
            P2 = np.array([float(x) for x in lines[2].split()[1:]]).reshape(3, 4)
            P3 = np.array([float(x) for x in lines[3].split()[1:]]).reshape(3, 4)
        return P2, P3

    def load_gt_labels(self, file_id):
        label_path = os.path.join(self.data_root, "label_2", f"{file_id}.txt")
        labels = []
        if not os.path.exists(label_path): return labels
        with open(label_path, 'r') as f:
            for line in f.readlines():
                d = line.split()
                if d[0] == 'DontCare': continue
                labels.append({
                    'type': d[0],
                    'h': float(d[8]), 'w': float(d[9]), 'l': float(d[10]),
                    'pos': (float(d[11]), float(d[12]), float(d[13])),
                    'ry': float(d[14]),
                    'bbox2d': [float(d[4]), float(d[5]), float(d[6]), float(d[7])]
                })
        return labels

    def run_eval(self):
        for file_id in self.file_ids:
            # 1. 加载数据
            img_l = cv2.imread(os.path.join(self.data_root, "image_2", f"{file_id}.png"))
            img_r = cv2.imread(os.path.join(self.data_root, "image_3", f"{file_id}.png"), cv2.IMREAD_GRAYSCALE)
            P2, P3 = self.parse_calib(file_id)

            # 2. 调用外部模块生成深度图
            depth_map = get_depth_map(cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY), img_r, P2, P3)

            # 3. 评估每个物体
            labels = self.load_gt_labels(file_id)
            print(f"\n--- {file_id} 评估结果 ---")

            for obj in labels:
                gt_z = obj['pos'][2]
                # 使用2D框确定采样区域
                x1, y1, x2, y2 = map(int, obj['bbox2d'])

                # 提取物体区域深度并取中位数
                roi_depth = depth_map[y1:y2, x1:x2]
                valid_mask = (roi_depth > 1.0) & (roi_depth < 80.0)
                if not np.any(valid_mask): continue

                pred_z = np.median(roi_depth[valid_mask])
                abs_err = abs(gt_z - pred_z)

                # 可视化
                cv2.rectangle(img_l, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_l, f"GT:{gt_z:.1f}m Pred:{pred_z:.1f}m", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                print(f"Type: {obj['type']:<8} | GT: {gt_z:>6.2f}m | Pred: {pred_z:>6.2f}m | Error: {abs_err:>5.2f}m")

            cv2.imshow("Evaluation Result", img_l)
            if cv2.waitKey(0) == 27: break

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(current_dir, "data", "object", "training")
    evaluator = KittiEvaluator(root)
    evaluator.run_eval()