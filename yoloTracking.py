import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO("yolo11s.pt")

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    result = model.track(frame, persist=True, tracker="botsort.yaml")
    print((result[0]).boxes.id)
    annotated_frame = result[0].plot()

    cv2.imshow("camera", annotated_frame)
    cv2.waitKey(1)
