from detect_face import extract_features
import cv2
import numpy as np

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def shear_image(image, shear_factor=0.2):
    height, width = image.shape[:2]
    M = np.float32([[1, shear_factor, 0], [shear_factor, 1, 0]])
    return cv2.warpAffine(image, M, (width, height))

def generate_features(detect_image):
    features = []
    transformations = {
        "original": detect_image,
        "flipped": cv2.flip(detect_image, 1),
        "rotated": cv2.warpAffine(detect_image, cv2.getRotationMatrix2D((detect_image.shape[1]//2, detect_image.shape[0]//2), 15, 1), (detect_image.shape[1], detect_image.shape[0])),
        "stretched": cv2.resize(cv2.resize(detect_image, (int(detect_image.shape[1] * 1.05), detect_image.shape[0])), (detect_image.shape[1], detect_image.shape[0])),
        "blurred": cv2.GaussianBlur(detect_image, (5,5), 0),
        "clahe": apply_clahe(detect_image),
        "sheared": shear_image(detect_image, 0.2)
    }
    
    for name, img in transformations.items():
        feature = extract_features(img)
        if feature is not None:
            features.append(feature)
    
    return features

