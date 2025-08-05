import cv2
import os

video_folder = r"AI_Learning\AIC"
output_folder = os.path.join(video_folder, "frames")
os.makedirs(output_folder, exist_ok=True)

def extract_frames_from_video(video_path, video_name, every_sec=2):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % (fps * every_sec) == 0:
            frame_name = f"{video_name}_frame_{saved:03d}.jpg"
            save_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(save_path, frame)
            saved += 1
        count += 1
    cap.release()


for file in os.listdir(video_folder):
    if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        video_path = os.path.join(video_folder, file)
        video_name = os.path.splitext(file)[0]
        print(f"Process: {file}")
        extract_frames_from_video(video_path, video_name)
        print(f"Finish: {file}")
