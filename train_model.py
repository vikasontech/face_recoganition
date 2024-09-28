import cv2
import numpy as np
from PIL import Image
import os
import pickle

def get_images_and_labels(dataset_path):
    image_paths = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                image_paths.append(os.path.join(root, file))
    
    face_samples = []
    ids = []
    label_map = {}
    current_id = 0
    
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    for image_path in image_paths:
        # Extract the label from the folder name
        label = os.path.basename(os.path.dirname(image_path))
        if label not in label_map:
            label_map[label] = current_id
            current_id += 1
        id_ = label_map[label]
        
        # Open the image and convert to grayscale
        pil_img = Image.open(image_path).convert('L')
        img_numpy = np.array(pil_img, 'uint8')
        
        # Detect faces in the image
        faces = face_detector.detectMultiScale(img_numpy)
        
        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id_)
    
    return face_samples, ids, label_map

def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    dataset_path = "dataset"
    faces, ids, label_map = get_images_and_labels(dataset_path)
    recognizer.train(faces, np.array(ids))
    recognizer.write("trainer.yml")
    
    # Save the label map
    with open("labels.pickle", 'wb') as f:
        pickle.dump(label_map, f)
    
    print("Training completed and trainer.yml saved.")

