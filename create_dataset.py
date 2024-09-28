import cv2
import os

def create_dataset(user_name, num_samples=100):
    # Create directory for the user
    dataset_path = f"dataset/{user_name}"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{dataset_path}/user.{count}.jpg", face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
            print(f"Image {count} saved for {user_name}")
        
        cv2.imshow('Capturing Faces', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count >= num_samples:
            break
    
    cap.release()
    cv2.destroyAllWindows()
