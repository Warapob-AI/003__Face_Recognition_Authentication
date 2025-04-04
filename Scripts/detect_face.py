from deepface import DeepFace
import numpy as np
import cv2 

prototxt_path = "003__Face_Recognition_Authentication/Pre-Trained/deploy.prototxt"
model_path = "003__Face_Recognition_Authentication/Pre-Trained/res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

def detect_face(image):
    h, w = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            padding = 20
            x1, y1, x2, y2 = max(0, x1 - padding), max(0, y1 - padding), min(w, x2 + padding), min(h, y2 + padding)

            face = image[y1:y2, x1:x2]
            return face 

    return None 

def extract_features(image):
    image = cv2.resize(image, (256, 256)) 
    
    try:
        features = DeepFace.represent(image, model_name="Facenet", enforce_detection=False)
        print("📌 Features Output:", features)
        
        if isinstance(features, list) and len(features) > 0:
            return np.array(features[0]['embedding'])
        else:
            print("⚠️ No embedding found")
            return None
    except Exception as e:
        print(f"⚠️ Error extracting features: {e}")
        return None