# # STEP 1: Import the necessary modules.
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import cv2
# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2
# import numpy as np


# def draw_landmarks_on_image(rgb_image, detection_result):
#   pose_landmarks_list = detection_result.pose_landmarks
#   annotated_image = np.copy(rgb_image)

#   # Loop through the detected poses to visualize.
#   for idx in range(len(pose_landmarks_list)):
#     pose_landmarks = pose_landmarks_list[idx]

#     # Draw the pose landmarks.
#     pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#     pose_landmarks_proto.landmark.extend([
#       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
#     ])
#     solutions.drawing_utils.draw_landmarks(
#       annotated_image,
#       pose_landmarks_proto,
#       solutions.pose.POSE_CONNECTIONS,
#       solutions.drawing_styles.get_default_pose_landmarks_style())
#   return annotated_image

# # STEP 2: Create an PoseLandmarker object.
# base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
# options = vision.PoseLandmarkerOptions(
#     base_options=base_options,
#     output_segmentation_masks=True)
# detector = vision.PoseLandmarker.create_from_options(options)

# # STEP 3: Load the input image.
# image = mp.Image.create_from_file("/home/xie/YOLOv8-pose/Dataset/000000000063.jpg") 
# #/home/xie/YOLOv8-pose/Dataset/right_augmented.jpg
# #/home/xie/YOLOv8-pose/Dataset/000000000063.jpg

# # STEP 4: Detect pose landmarks from the input image.
# detection_result = detector.detect(image)

# # STEP 5: Process the detection result. In this case, visualize it.
# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)


# # Convert the annotated image from RGB to BGR format (OpenCV uses BGR)
# annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

# # Specify the output path where you want to save the image
# output_path = "/home/xie/YOLOv8-pose/Dataset/origin_annotated.jpg"

# # Save the image
# cv2.imwrite(output_path, annotated_image_bgr)

# print(f"Annotated image saved to: {output_path}")




# ## test on event camera
# import os
# import cv2
# import torch
# import numpy as np
# from utils import util
# from nets import nn


# @torch.no_grad()
# def run_pose_estimation_and_save_images(input_path, output_dir, input_size=640, conf_thres=0.25, iou_thres=0.7):
#     os.makedirs(output_dir, exist_ok=True)

#     # Load model
#     checkpoint = torch.load('./weights/best.pt', weights_only=False, map_location='cuda')
#     model = checkpoint['model'].float().cuda()
#     model.eval()
#     model.half()

#     stride = int(max(model.stride.cpu().numpy()))
#     kpt_shape = model.head.kpt_shape

#     # Video IO
#     cap = cv2.VideoCapture(input_path)
#     assert cap.isOpened(), f"Cannot open video {input_path}"

#     # Color palette and skeleton
#     palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0],
#                         [255, 153, 255], [153, 204, 255], [255, 102, 255], [255, 51, 255],
#                         [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102],
#                         [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51],
#                         [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8)

#     skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
#                 [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
#                 [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

#     kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
#     limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

#     idx = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         orig = frame.copy()
#         shape = frame.shape[:2]
#         r = min(1.0, input_size / shape[0], input_size / shape[1])
#         new_shape = int(shape[1] * r), int(shape[0] * r)
#         resized = cv2.resize(frame, new_shape)
#         padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
#         padded[:new_shape[1], :new_shape[0]] = resized

#         img = padded.transpose(2, 0, 1)[::-1]  # HWC to CHW, BGR to RGB
#         img = np.ascontiguousarray(img)
#         img = torch.from_numpy(img).unsqueeze(0).half().cuda() / 255.0

#         # Inference
#         pred = model(img)
#         pred = util.non_max_suppression(pred, conf_thres, iou_thres, model.head.nc)

#         for det in pred:
#             if det is None or not len(det):
#                 continue
#             det = det.clone()
#             box_output = det[:, :6]
#             kps_output = det[:, 6:].view(len(det), *kpt_shape)

#             gain = r
#             pad = (input_size - new_shape[0], input_size - new_shape[1])  # width, height padding

#             box_output[:, [0, 2]] -= pad[0] / 2
#             box_output[:, [1, 3]] -= pad[1] / 2
#             box_output[:, :4] /= gain
#             box_output[:, [0, 2]] = box_output[:, [0, 2]].clamp(0, shape[1])
#             box_output[:, [1, 3]] = box_output[:, [1, 3]].clamp(0, shape[0])

#             kps_output[..., 0] -= pad[0] / 2
#             kps_output[..., 1] -= pad[1] / 2
#             kps_output[..., :2] /= gain
#             kps_output[..., 0].clamp_(0, shape[1])
#             kps_output[..., 1].clamp_(0, shape[0])

#             for box in box_output:
#                 x1, y1, x2, y2 = map(int, box[:4])
#                 cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             for kpt in kps_output:
#                 for i, (x, y, *rest) in enumerate(kpt):
#                     if len(rest) and rest[0] < 0.5:
#                         continue
#                     color = [int(c) for c in kpt_color[i]]
#                     cv2.circle(orig, (int(x), int(y)), 3, color, -1, lineType=cv2.LINE_AA)

#                 for i, (p1, p2) in enumerate(skeleton):
#                     x1, y1, c1 = kpt[p1 - 1][:3]
#                     x2, y2, c2 = kpt[p2 - 1][:3]
#                     if c1 < 0.5 or c2 < 0.5:
#                         continue
#                     cv2.line(orig, (int(x1), int(y1)), (int(x2), int(y2)),
#                              [int(c) for c in limb_color[i]], 2, lineType=cv2.LINE_AA)

#         # Save image
#         out_path = os.path.join(output_dir, f"frame_{idx:05d}.jpg")
#         cv2.imwrite(out_path, orig)
#         idx += 1

#     cap.release()
#     print(f"Saved {idx} frames to {output_dir}")


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input', type=str, required=True, help='Path to input .webm video')
#     parser.add_argument('--output-dir', type=str, default='./results', help='Directory to save output images')
#     parser.add_argument('--input-size', type=int, default=640, help='Input size for model')
#     args = parser.parse_args()

#     run_pose_estimation_and_save_images(args.input, args.output_dir, args.input_size)


# ## Test on event camera(mediapipe)
# import os
# import cv2
# import mediapipe as mp
# import numpy as np
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2


# def draw_landmarks_on_image(rgb_image, detection_result):
#     pose_landmarks_list = detection_result.pose_landmarks
#     annotated_image = np.copy(rgb_image)

#     for idx in range(len(pose_landmarks_list)):
#         pose_landmarks = pose_landmarks_list[idx]

#         pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#         pose_landmarks_proto.landmark.extend([
#             landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
#             for landmark in pose_landmarks
#         ])

#         solutions.drawing_utils.draw_landmarks(
#             annotated_image,
#             pose_landmarks_proto,
#             solutions.pose.POSE_CONNECTIONS,
#             solutions.drawing_styles.get_default_pose_landmarks_style())

#     return annotated_image


# def process_video(video_path, output_dir, model_path='pose_landmarker.task'):
#     os.makedirs(output_dir, exist_ok=True)

#     # Create PoseLandmarker
#     base_options = python.BaseOptions(model_asset_path=model_path)
#     options = vision.PoseLandmarkerOptions(
#         base_options=base_options,
#         output_segmentation_masks=False)  # Set True if needed
#     detector = vision.PoseLandmarker.create_from_options(options)

#     # Open video
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise FileNotFoundError(f"Cannot open video: {video_path}")

#     frame_idx = 0
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         # Convert BGR to RGB
#         rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

#         # Pose detection
#         detection_result = detector.detect(mp_image)

#         # Draw pose landmarks
#         annotated_image = draw_landmarks_on_image(rgb_image, detection_result)

#         # Convert back to BGR for saving
#         annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

#         # Save frame
#         output_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
#         cv2.imwrite(output_path, annotated_bgr)
#         frame_idx += 1

#     cap.release()
#     print(f"Saved {frame_idx} annotated frames to: {output_dir}")


# if __name__ == '__main__':
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input', type=str, required=True, help='Path to input .webm video')
#     parser.add_argument('--output-dir', type=str, default='./output_mediapipe', help='Output folder for annotated frames')
#     parser.add_argument('--model', type=str, default='pose_landmarker.task', help='Path to .task model')
#     args = parser.parse_args()

#     process_video(args.input, args.output_dir, args.model)


# ## Test official yolo pose
# from ultralytics import YOLO
# import cv2
# import os

# # Load the model
# model = YOLO("/home/xie/YOLOv8-pose/ultralytics/yolov8x-pose.pt")

# # Create output directory if it doesn't exist
# os.makedirs("./output", exist_ok=True)

# # Predict on an image
# # results = model("https://ultralytics.com/images/bus.jpg")
# results = model("../Dataset/coco8-pose/images/val/000000000113.jpg")  # Change to your image path
# print(results)

# # Loop through results and save visualized predictions
# for i, result in enumerate(results):
#     # Get the image with keypoints drawn
#     plotted_img = result.plot()

#     # Save the image
#     save_path = f"./output/result_{i}.jpg"
#     cv2.imwrite(save_path, cv2.cvtColor(plotted_img, cv2.COLOR_RGB2BGR))
#     print(f"Saved result to {save_path}")



import numpy as np
import cv2
import os

def plot_keypoints_on_image(image_path, txt_path, output_suffix="_labeled"):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    img_height, img_width = image.shape[:2]
    
    # Read keypoints from text file
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    # Process each detection (each line in the text file)
    for line in lines:
        data = list(map(float, line.strip().split()))
        if len(data) < 5:  # Skip invalid lines
            continue
            
        # Extract bbox (class, x_center, y_center, width, height)
        class_id, x_center, y_center, width, height = data[:5]
        
        # Extract keypoints (17 keypoints, each with x, y, visibility)
        keypoints = np.array(data[5:]).reshape(-1, 3)  # Shape: (17, 3)
        
        # COCO keypoint names (in order)
        coco_keypoints = [
            "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
            "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
            "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
            "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
        ]
        
        # Convert normalized coordinates to absolute coordinates
        def norm_to_abs(x_norm, y_norm):
            return int(x_norm * img_width), int(y_norm * img_height)
        
        # Draw bounding box
        bbox_x1 = (x_center - width/2) * img_width
        bbox_y1 = (y_center - height/2) * img_height
        bbox_x2 = (x_center + width/2) * img_width
        bbox_y2 = (y_center + height/2) * img_height
        cv2.rectangle(image, (int(bbox_x1), int(bbox_y1)), 
                      (int(bbox_x2), int(bbox_y2)), (0, 255, 0), 2)
        
        # Draw keypoints and labels
        for i, (x, y, v) in enumerate(keypoints):
            if v > 0:  # Only plot visible keypoints
                abs_x, abs_y = norm_to_abs(x, y)
                # Draw keypoint
                cv2.circle(image, (abs_x, abs_y), 5, (0, 0, 255), -1)
                # Draw keypoint label
                cv2.putText(image, coco_keypoints[i], (abs_x + 10, abs_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Save the output image
    base_name = os.path.splitext(image_path)[0]
    output_path = f"/home/xie/YOLOv8-pose/Dataset/output/{output_suffix}.jpg"
    cv2.imwrite(output_path, image)
    print(f"Saved result to {output_path}")

# Example usage:
image_path = "Dataset/coco-pose/images/train2017/000000378118.jpg"
txt_path = "Dataset/coco-pose/labels/train2017/000000378118.txt"
plot_keypoints_on_image(image_path, txt_path)
