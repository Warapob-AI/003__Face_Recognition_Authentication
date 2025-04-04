from data_augmentation import generate_features
from detect_face import detect_face
from numpy import dot
from numpy.linalg import norm
import numpy as np
import sqlite3
import cv2

def login_face(firstname, lastname, image_path, db_path):
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

        cursor.execute("SELECT embedding FROM users WHERE firstname = ? AND lastname = ?", (firstname, lastname))
        user_data = cursor.fetchall()
        
        conn.close()
        
        if not user_data:
            print("ไม่พบข้อมูลผู้ใช้ในระบบ")
            return False
        
        db_embedding = user_data[0][0]  
        db_embedding_np = np.frombuffer(db_embedding, dtype=np.float64)

        cosine_similarity = dot(final_vector, db_embedding_np) / (norm(final_vector) * norm(db_embedding_np))
        print("Cosine Similarity:", cosine_similarity)

        if cosine_similarity > 0.6: 
            print(f"เข้าสู่ระบบสำเร็จ! ยินดีต้อนรับ {firstname} {lastname}")
            return True
        else:
            print("ไม่สามารถเข้าสู่ระบบได้: ใบหน้าไม่ตรงกับข้อมูลในระบบ")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False
