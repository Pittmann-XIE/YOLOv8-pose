import cv2
import os

def extract_frames_with_options(mp4_path, output_dir="/home/xie/YOLOv8-pose/Dataset/human_event/val/images", txt_file="/home/xie/YOLOv8-pose/Dataset/human_event/frame_paths.txt", 
                               frame_interval=1, max_frames=None):
    """
    Extract frames from MP4 with additional options
    
    Args:
        mp4_path: Path to the MP4 file
        output_dir: Directory to save extracted frames
        txt_file: Path to txt file that will contain image paths
        frame_interval: Extract every Nth frame (1 = all frames, 10 = every 10th frame)
        max_frames: Maximum number of frames to extract (None = all frames)
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(txt_file), exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(mp4_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {mp4_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS")
    
    frame_count = 0
    saved_count = 0
    frame_paths = []
    
    print("Extracting frames...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Check if we should save this frame based on interval
        if frame_count % frame_interval == 0:
            # Create frame filename
            frame_filename = f"frame_{saved_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            # Save frame
            cv2.imwrite(frame_path, frame)
            
            # Store path for txt file
            frame_paths.append(frame_path)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"Saved {saved_count} frames...")
            
            # Check if we've reached max frames limit
            if max_frames and saved_count >= max_frames:
                break
        
        frame_count += 1
    
    # Release video capture
    cap.release()
    
    # Write paths to txt file
    with open(txt_file, 'w') as f:
        for path in frame_paths:
            f.write(path + '\n')
    
    print(f"Extraction complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames saved: {saved_count}")
    print(f"Frames saved in: {output_dir}")
    print(f"Frame paths saved in: {txt_file}")

# Usage examples
if __name__ == "__main__":
    video_path = "/home/xie/YOLOv8-pose/Screencast from 04.08.2025 17_26_05.webm"
    
    # Extract all frames
    extract_frames_with_options(video_path)
    
    # Extract every 10th frame
    # extract_frames_with_options(video_path, frame_interval=10)
    
    # Extract maximum 1000 frames
    # extract_frames_with_options(video_path, max_frames=1000)