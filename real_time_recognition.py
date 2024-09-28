import cv2
import numpy as np
import pickle

def load_label_map():
    with open("labels.pickle", 'rb') as f:
        label_map = pickle.load(f)
    # Reverse the label map to get names from IDs
    labels = {v: k for k, v in label_map.items()}
    return labels

def load_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")
    labels = load_label_map()
    
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_detector, recognizer

def recognize_faces_from_webcam():
    # face_detector, recognizer =  load_faces()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")
    labels = load_label_map()
    
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(face)
            if conf < 50:
                name = labels.get(id_, "Unknown")
            else:
                name = "Unknown"
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Display name
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        
        cv2.imshow('Facial Recognition', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     recognize_faces()

# def check_face_in_data_set(recognizer, faces, labels,  gray):
   

def image_check(image_path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")
    labels = load_label_map()
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # face_detector =  load_faces()
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

    # Detect faces in the input image
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    print(f"Detected {len(faces)} face(s).")

    for (x, y, w, h) in faces:
        # Get the region of interest (the face)
        roi_gray = gray[y:y+h, x:x+w]

        # Predict the label for the detected face
        label_id, confidence = recognizer.predict(roi_gray)

        if confidence < 100:  # Lower confidence is better in OpenCV's LBPH
            print(f"Recognized {labels[label_id]} with confidence {confidence}")
            
            # Draw a rectangle around the face and put the label
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image, labels[label_id], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            print("Unknown face")
            # If the face is not recognized, draw a rectangle and label it as 'Unknown'
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(image, 'Unknown', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the image with rectangles and labels
        print(f"Image found: {image}")
        # cv2.imshow("Faces Found", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()




