import os
import cv2
import torch
import numpy as np
from utils import util
from nets import nn
from glob import glob
from tqdm import tqdm


@torch.no_grad()
def run_pose_estimation_on_images(input_folder, output_folder, input_size=640, conf_thres=0.25, iou_thres=0.7):
    os.makedirs(output_folder, exist_ok=True)

    # Load model
    checkpoint = torch.load('./weights/best.pt', weights_only=False, map_location='cuda')
    model = checkpoint['model'].float().cuda()
    model.eval()
    model.half()

    stride = int(max(model.stride.cpu().numpy()))
    kpt_shape = model.head.kpt_shape

    # Skeleton and color setup
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0],
                        [255, 153, 255], [153, 204, 255], [255, 102, 255], [255, 51, 255],
                        [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51],
                        [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8)

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
                [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

    # Process each image
    image_paths = sorted(glob(os.path.join(input_folder, '*.*')))
    for path in tqdm(image_paths, desc="Processing images"):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(f"Could not load {path}")
            continue
        orig = img_bgr.copy()
        shape = img_bgr.shape[:2]

        r = min(1.0, input_size / shape[0], input_size / shape[1])
        new_shape = int(shape[1] * r), int(shape[0] * r)
        resized = cv2.resize(img_bgr, new_shape)
        padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        padded[:new_shape[1], :new_shape[0]] = resized

        img = padded.transpose(2, 0, 1)[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).unsqueeze(0).half().cuda() / 255.0

        # Inference
        pred = model(img)
        pred = util.non_max_suppression(pred, conf_thres, iou_thres, model.head.nc)

        for det in pred:
            if det is None or not len(det):
                continue
            det = det.clone()

            box_output = det[:, :6]
            kps_output = det[:, 6:].view(len(det), *kpt_shape)

            gain = r
            pad = (input_size - new_shape[0], input_size - new_shape[1])  # width, height padding

            box_output[:, [0, 2]] -= pad[0] / 2
            box_output[:, [1, 3]] -= pad[1] / 2
            box_output[:, :4] /= gain
            box_output[:, [0, 2]] = box_output[:, [0, 2]].clamp(0, shape[1])
            box_output[:, [1, 3]] = box_output[:, [1, 3]].clamp(0, shape[0])

            kps_output[..., 0] -= pad[0] / 2
            kps_output[..., 1] -= pad[1] / 2
            kps_output[..., :2] /= gain
            kps_output[..., 0].clamp_(0, shape[1])
            kps_output[..., 1].clamp_(0, shape[0])

            for box in box_output:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for kpt in kps_output:
                for i, (x, y, *rest) in enumerate(kpt):
                    if len(rest) and rest[0] < 0.5:
                        continue
                    color = [int(c) for c in kpt_color[i]]
                    cv2.circle(orig, (int(x), int(y)), 3, color, -1, lineType=cv2.LINE_AA)

                for i, (p1, p2) in enumerate(skeleton):
                    x1, y1, c1 = kpt[p1 - 1][:3]
                    x2, y2, c2 = kpt[p2 - 1][:3]
                    if c1 < 0.5 or c2 < 0.5:
                        continue
                    cv2.line(orig, (int(x1), int(y1)), (int(x2), int(y2)),
                             [int(c) for c in limb_color[i]], 2, lineType=cv2.LINE_AA)

        save_name = os.path.basename(path)
        save_path = os.path.join(output_folder, save_name)
        cv2.imwrite(save_path, orig)

    print(f"Saved pose estimation results to '{output_folder}'")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='input', help='Input image folder')
    parser.add_argument('--output', type=str, default='output', help='Output folder to save results')
    parser.add_argument('--input-size', type=int, default=640, help='Model input size')
    args = parser.parse_args()

    run_pose_estimation_on_images(args.input, args.output, args.input_size)
