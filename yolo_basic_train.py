import ultralytics
from ultralytics import YOLO
from glob import glob

img_list = glob(r'/train/images/*.jpg') # train image path
val_img_list = glob(r'/test/images/*.jpg') # test image path

with open(r'/train.txt', 'w') as f:
    f.write('\n'.join(img_list) + '\n')

with open(r'/test.txt', 'w') as f:
    f.write('\n'.join(val_img_list) + '\n')

data_location = r'/data.yaml'
weight_location = r'/weight_location'

# Load a model
model = YOLO(r'yolov8n.pt')  # load a pretrained model (recommended for training)
model.train(data=data_location, project=weight_location, epochs=10)