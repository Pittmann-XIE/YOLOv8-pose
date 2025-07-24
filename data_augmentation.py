import cv2
import numpy as np
import os

def augment(image, alpha, beta, save_prefix="augmented"):
    img = image.copy()

    height, width = img.shape[:2]
    print(f'Image dimensions: {height}x{width}')
    mid_x = width // 2 - 150

    # Split left and right parts
    left_half = img[:, :mid_x]
    right_half = img[:, mid_x:]

    # Apply brightness/contrast to the left half
    right_aug = cv2.convertScaleAbs(right_half, alpha=1+alpha, beta=beta)
    left_aug = cv2.convertScaleAbs(left_half, alpha=alpha, beta=beta)

    # Combine modified left and original right
    augmented = np.hstack((left_aug, right_aug))

    # Save result
    save_path = f"/home/xie/YOLOv8-pose/Dataset/{save_prefix}.jpg"
    success = cv2.imwrite(save_path, augmented)
    if not success:
        raise RuntimeError(f"Failed to save image to: {save_path}")

    return augmented

def main():
    input_path = '/home/xie/YOLOv8-pose/Dataset/images/test2017/000000000063.jpg'
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Image not found: {input_path}")

    img = cv2.imread(input_path)

    augment(
        image=img,
        alpha=0.1,  # Contrast adjustment
        beta=0,    # Brightness increase
        save_prefix="right_augmented_more"
    )

if __name__ == "__main__":
    main()
