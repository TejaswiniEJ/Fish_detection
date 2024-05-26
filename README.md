#import YOLOV8 utilies
!pip install ultralytics
!pip install supervision
from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw
import requests
import supervision as sv

#import fish datset
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="tLlJqdTxCDmaOESbEZ2C")
project = rf.workspace("1253971414qqcom-gpnwq").project("new_fish_detect_2")
version = project.version(2)
dataset = version.download("yolov8")

#set path
model.train(data="/content/new_fish_detect_2-2/data.yaml",epochs=5,conf=0.7)
#upload live vedio
trained_model = YOLO('/content/runs/detect/train/weights/best.pt')
from google.colab import files
uploaded = files.upload()
list(uploaded.keys())[0]
out = trained_model.predict(list(uploaded.keys())[0],save=True)
