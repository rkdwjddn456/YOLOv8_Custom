import ultralytics
from ultralytics import YOLO
import cv2

model_path = r'/best.pt'

# Load a model
model = YOLO(model_path)

conf_thresh = 0.5 # Set the confidence threshold
image_path = r'/test/images/test_image.jpg'
image_save_path = r'/save_image'

im = cv2.imread(image_path)

results = model.predict(source=im, save=True, project=image_save_path, save_crop=True, conf=conf_thresh)