import cv2
import requests
import numpy as np
from io import BytesIO
from datetime import datetime
from function.logic.faceCam.faceFilter import FaceFilter
import face_recognition

def upload_and_send_metadata(face_img, name="default_name", camera_id="cam_01"):
    try:
        success, encoded_image = cv2.imencode(".jpg", face_img)
        if not success:
            print("[ERROR] Failed to encode image")
            return

        image_bytes = BytesIO(encoded_image.tobytes())

        files = {'images': ('face.jpg', image_bytes, 'image/jpeg')}
        data = {'camera_id': camera_id}
        res = requests.post("http://localhost:8889/api/UploadImageAndReturnPath", files=files, data=data)
        uploaded_paths = res.json().get("paths", [])

        if not uploaded_paths:
            print("[ERROR] No image path returned from Go")
            return

        image_path = uploaded_paths[0]

        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        if not encodings:
            print("[ERROR] No face encoding found")
            return

        metadata = {
            "name": name,
            "camera_id": camera_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "embedding": encodings[0].tolist(),
            "image_path": image_path,
            "confidence": 0.95
        }

        meta_res = requests.post("http://localhost:8889/api/InputFaceRecognitionDetails", json=metadata)

        if meta_res.status_code == 200:
            print(f"[INFO] Upload + Metadata OK: {image_path}")
        else:
            print(f"[ERROR] Metadata failed: {meta_res.status_code} {meta_res.text}")

    except Exception as e:
        print(f"[ERROR] {e}")
