import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import torch
import timm
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from skimage import io
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DATA_DIR = "/home/akansh_26/Hackathons/VisionX-AI/Person-Re-Id-Dataset/train"
DEVICE = "cuda"

MAIN_ENCODING = {}
# STATE = "PICK" # PICK / TRACK
# user_input = None
count = 0
URL1 = "./vid1.mp4"
URL2 = "./vid2.mp4"


model_yolo1 = YOLO("yolo11s.pt")
cam1 = cv2.VideoCapture(URL1)

model_yolo2 = YOLO("yolo11s.pt")
cam2 = cv2.VideoCapture(URL2)


def main():
    # ret1, frame1 = cam1.read()
    # ret2, frame2 = cam2.read()

    # cv2.imshow("Camera 1", frame1)
    # cv2.imshow("Camera 2", frame2)
    # camid = input("Enter which cam you want to select: ")
    
    while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()

        if not ret1 or not ret2:
            break

        annotated_frame1 = process_frame(frame1, model_yolo1, 1)
        annotated_frame2 = process_frame(frame2, model_yolo1, 2)

        cv2.imshow("frame1", annotated_frame1)
        cv2.imshow("frame2", annotated_frame2)

        cv2.waitKey(1)
    cv2.destroyAllWindows()


def process_frame(frame, model_yolo, camera_number):
    global count
    global MAIN_ENCODING

    dist = 1000000

    results = model_yolo.track(frame, persist=True, tracker="botsort.yaml")

    boxes = results[0].boxes.xywh.cpu().numpy()  # Format: [x, y, width, height]
    confs = results[0].boxes.conf.cpu().numpy()   # Confidence scores
    classes = results[0].boxes.cls.cpu().numpy()    # Class labels (as indices)
    track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else None

    threshold = 4

    if camera_number == 1:

        if boxes is not None and boxes.any():
            count += 1

        if count == 1:
            for i, box in enumerate(boxes):
                cls = int(classes[i])
                if cls != 0:
                    continue

                tid = int(track_ids[i]) if track_ids is not None else -1
                conf = confs[i]

                x, y, w, h = box
                pt1 = (int(x-w//2), int(y-h//2))
                pt2 = (int(x+w//2), int(y+h//2))

                # if tid == 1 and -1 < count < 100 and camera_number == 1:
                #     img_person = frame[int(x-w//2):int(x+w//2), int(y-h//2):int(y+h//2)]
                #     img_person = torch.from_numpy(img_person).permute(2, 0, 1) / 255.0
                #     img_person = img_person.to(DEVICE)
                #     MAIN_ENCODING = model_siamese(img_person.unsqueeze(0))
                #     MAIN_ENCODING = MAIN_ENCODING.detach().cpu().numpy()
                #     print(f"Enc ID-1: {MAIN_ENCODING}")

                img_person = frame[int(y-h//2):int(y+h//2), int(x-w//2):int(x+w//2)]
                img_person = cv2.resize(img_person, (128, 64))
                img_person = torch.from_numpy(img_person).permute(2, 0, 1) / 255.0
                model_siamese.eval()
                with torch.no_grad():
                    img_person = img_person.to(DEVICE)
                    person_enc = model_siamese(img_person.unsqueeze(0))
                    person_enc = person_enc.detach().cpu().numpy()

                MAIN_ENCODING[tid] = person_enc
        
        else:
            
            for i, box in enumerate(boxes):
                cls = int(classes[i])
                if cls != 0:
                    continue

                tid = int(track_ids[i]) if track_ids is not None else -1
                conf = confs[i]

                x, y, w, h = box
                pt1 = (int(x-w//2), int(y-h//2))
                pt2 = (int(x+w//2), int(y+h//2))

                img_person = frame[int(y-h//2):int(y+h//2), int(x-w//2):int(x+w//2)]
                img_person = cv2.resize(img_person, (128, 64))
                img_person = torch.from_numpy(img_person).permute(2, 0, 1) / 255.0
                model_siamese.eval()
                with torch.no_grad():
                    img_person = img_person.to(DEVICE)
                    person_enc = model_siamese(img_person.unsqueeze(0))
                    person_enc = person_enc.detach().cpu().numpy()

                distances = []
                ids_list = []
                for i in MAIN_ENCODING:
                    dist = euclidean_dist(MAIN_ENCODING[i], person_enc)
                    if dist < threshold:
                        distances.append(dist)
                        ids_list.append(i)
                
                try:
                    idx = np.argmin(distances)
                    cv2.putText(frame, f"ID:{ids_list[idx]}", (int(x-w//2), int(y-h//2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except:
                    pass
                
                cv2.rectangle(frame, pt1, pt2, color=(0, 255, 0), thickness=2)
    else:
        for i, box in enumerate(boxes):
            cls = int(classes[i])
            if cls != 0:
                continue

            tid = int(track_ids[i]) if track_ids is not None else -1
            conf = confs[i]

            x, y, w, h = box
            pt1 = (int(x-w//2), int(y-h//2))
            pt2 = (int(x+w//2), int(y+h//2))

            img_person = frame[int(y-h//2):int(y+h//2), int(x-w//2):int(x+w//2)]
            img_person = cv2.resize(img_person, (128, 64))
            img_person = torch.from_numpy(img_person).permute(2, 0, 1) / 255.0
            model_siamese.eval()
            with torch.no_grad():
                img_person = img_person.to(DEVICE)
                person_enc = model_siamese(img_person.unsqueeze(0))
                person_enc = person_enc.detach().cpu().numpy()

            distances = []
            ids_list = []
            for i in MAIN_ENCODING:
                dist = euclidean_dist(MAIN_ENCODING[i], person_enc)
                if dist < threshold:
                    distances.append(dist)
                    ids_list.append(i)
            
            try:
                idx = np.argmin(distances)
                cv2.putText(frame, f"ID:{ids_list[idx]}", (int(x-w//2), int(y-h//2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except:
                pass

            cv2.rectangle(frame, pt1, pt2, color=(0, 255, 0), thickness=2)
            

    return frame


class APN_Model(nn.Module):

    def __init__(self, emb_size=512):
        super(APN_Model, self).__init__()

        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
        self.efficientnet.classifier = nn.Linear(in_features=self.efficientnet.classifier.in_features, out_features=emb_size)

    def forward(self, images):

        embeddings = self.efficientnet(images)
        return embeddings


model_siamese = APN_Model()
model_siamese.to(DEVICE)
model_siamese.load_state_dict(torch.load("best_model.pt"))


def get_input():
    global user_input
    user_input = input("Enter a number: ")

def euclidean_dist(enc1, enc2):
    dist = np.sqrt(np.dot(enc1 - enc2, (enc1 - enc2).T))
    return dist


if __name__ == "__main__":
    main()
