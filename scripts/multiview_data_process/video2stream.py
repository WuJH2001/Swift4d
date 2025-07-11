import cv2
import os
from argparse import ArgumentParser

# parser = ArgumentParser("Video converter")
# parser.add_argument("--source_path", "-s", required=True, type=str)
# args = parser.parse_args()

def extract_frames_to_folders(video_path, base_folder, video_label):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_folder = f"{base_folder}{frame_idx:06d}"
        os.makedirs(frame_folder, exist_ok=True)
        output_filename = f"{video_label}.png"
        output_path = os.path.join(frame_folder, output_filename)
        cv2.imwrite(output_path, frame)
        print(f"Saved {output_path}")
        frame_idx += 1
    cap.release()




for i in range(0,34):  # Total videos you want to process
    frame_id = f'{i:02d}'
    video_filename = os.path.join("vru", frame_id+".mp4")  # Maybe you need to modify the file input path to match your data.
    base_folder = "vru"+ "/frame"
    video_label = f"cam{i:02d}"
    extract_frames_to_folders(video_filename, base_folder, video_label)
