import face_recognition
import numpy as np
from PIL import Image
import io
import cv2
import dlib
import os

# Load dlib's facial landmark predictor
PREDICTOR_PATH = os.path.join(os.path.dirname(__file__), "assets", "shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

# Eye landmark indices for EAR
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

def get_embedding(img_bytes):
    """
    Convert image bytes to face embedding vector using face_recognition
    """
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image_np = np.array(image)

    # Detect face
    face_locations = face_recognition.face_locations(image_np)
    if not face_locations:
        return None

    embeddings = face_recognition.face_encodings(image_np, face_locations)
    return embeddings[0] if embeddings else None

def detect_blink(frame, threshold=0.2):
    """
    Detect blink using eye aspect ratio (EAR).
    Returns True if EAR is below threshold.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        def eye_aspect_ratio(eye):
            A = np.linalg.norm(eye[1] - eye[5])
            B = np.linalg.norm(eye[2] - eye[4])
            C = np.linalg.norm(eye[0] - eye[3])
            return (A + B) / (2.0 * C)

        left_ear = eye_aspect_ratio(landmarks[LEFT_EYE])
        right_ear = eye_aspect_ratio(landmarks[RIGHT_EYE])
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < threshold:
            return True

    return False
