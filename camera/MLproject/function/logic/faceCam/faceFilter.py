import numpy as np
import cv2
import face_recognition

class FaceFilter:
    def __init__(self, threshold=0.6):
        self.last_embedding = None
        self.threshold = threshold
        print(f"[DEBUG] Threshold used by FaceFilter = {self.threshold}")

    def is_different(self, face_img):
        """Original method that accepts image"""
        if face_img is None or len(face_img.shape) != 3 or face_img.shape[2] != 3:
            print("[ERROR] Invalid face image input")
            return False

        if face_img.dtype != np.uint8:
            face_img = np.clip(face_img, 0, 255).astype(np.uint8)

        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)

        try:
            encodings = face_recognition.face_encodings(rgb)
        except Exception as e:
            print(f"[ERROR] face_encodings failed: {e}")
            return False

        if not encodings:
            print("[WARNING] No face encoding found")
            return False

        return self._compare_encoding(encodings[0])

    def is_different_with_encoding(self, current_encoding):
        """Optimized method that accepts pre-computed encoding"""
        return self._compare_encoding(current_encoding)

    def _compare_encoding(self, current_encoding):
        """Internal comparison logic"""
        if self.last_embedding is None:
            self.last_embedding = current_encoding
            print("[FaceFilter] First face captured")
            return True

        distance = np.linalg.norm(current_encoding - self.last_embedding)
        print(f"[INFO] Face distance: {distance:.4f}")

        if distance > self.threshold:
            self.last_embedding = current_encoding
            return True

        return False