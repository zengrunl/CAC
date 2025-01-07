import cv2
import numpy as np
import os
from tqdm import tqdm  # For progress bar, optional

src_dir = "/root/data1/LSR/LSR2/lrs2_rebuild/faces/"
dst_dir = "/root/data1/LSR/LSR2/lrs2_rebuild/face_flow/"
os.makedirs(dst_dir, exist_ok=True)

video_files = [f for f in os.listdir(src_dir) if f.endswith('.mp4')]
# video_files = video_files[:20392]  # Keep only the first 20,000 files

for video_file in tqdm(video_files, desc="Calculating optical flows"):
    video_path = os.path.join(src_dir, video_file)
    flow_path = os.path.join(dst_dir, video_file.replace('.mp4', '.npy'))

    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Create a list to store flow data
    flow_data = []

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev=prev_gray, next=next_gray,
            flow=None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        flow_data.append(flow)

        # Update the previous frame
        prev_gray = next_gray

    cap.release()

    # Save the flow data as a NumPy file
    np.save(flow_path, np.array(flow_data))

print("Optical flow calculation and saving completed.")