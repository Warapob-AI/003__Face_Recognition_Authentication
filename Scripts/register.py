import numpy as np
import sqlite3
import cv2
import sys
import os 
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data_augmentation import generate_features
from detect_face import detect_face


def register_face(firstname, lastname, image_path, db_path):
    features = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        image = cv2.imread(image_path)
        detect_image = detect_face(image)
        
        if detect_image is None:
            print('ไม่พบใบหน้า')
            return False
        
        features = generate_features(detect_image)

        vectorizer = np.array(features)
        final_vector = np.mean(vectorizer, axis=0) 
        
        print(final_vector)

        cursor.execute("INSERT INTO users (firstname, lastname, embedding) VALUES (?, ?, ?)", 
                           (firstname, lastname, final_vector.tobytes()))

        conn.commit()
        conn.close()
        print("ลงทะเบียนสำเร็จพร้อมการขยายข้อมูล!")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
