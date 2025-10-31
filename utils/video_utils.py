import cv2 

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        # ret is False when there are no more frames to read
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# Function to save video frames to a video file
def save_video(out_video_frame, output_path):
    if len(out_video_frame) == 0:
        print("No frames to save.")
        return
    height, width, layers = out_video_frame[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Be sure to use lower case
    video = cv2.VideoWriter(output_path, fourcc, 24, (width, height))
    for frame in out_video_frame:
        video.write(frame)
    video.release()