import threading
import cv2
from ultralytics import YOLO

def run_tracker_in_thread(filename, model, crop_path, file_index):
    video = cv2.VideoCapture(filename)  # Read the video file

    while True:
        ret, frame = video.read()  # Read the video frames

        # Exit the loop if no more frames in either video
        if not ret:
            break

        # Track objects in frames if available
        results = model.track(frame, persist=True, save_crop=True, project=crop_path)
        res_plotted = results[0].plot()
        cv2.imshow(f"Tracking_Stream_{file_index}", res_plotted)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release video sources
    video.release()

# Load a model
model_path = r'/train/weights/best.pt'

model = YOLO(model_path)

# Define the video files for the trackers
video_file = r'/video/Shuka.mp4'

# Define the crop images path
crop_path = r'/video/crop_image'

# Create the tracker threads
tracker_thread = threading.Thread(target=run_tracker_in_thread, args=(video_file, model, crop_path, 1), daemon=True)

# Start the tracker threads
tracker_thread.start()

# Wait for the tracker threads to finish
tracker_thread.join()

# Clean up and close windows
cv2.destroyAllWindows()