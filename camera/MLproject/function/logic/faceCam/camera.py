import cv2
import time
import numpy as np
import requests
import joblib
from io import BytesIO
from datetime import datetime
from function.logic.faceCam.faceFilter import FaceFilter
import face_recognition
from pathlib import Path
import joblib

class CameraApp:
    def __init__(self, detector, cam_index=1, save_interval=2, mode="recognize"):
    
        self.cap = cv2.VideoCapture(cam_index)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Camera FPS: {fps}")
        self.mode = mode # adding mode parameter from main.py

        self.detector = detector
        self.face_filter = FaceFilter(threshold=0.6)
        self.save_interval = save_interval
        self.last_checked_time = 0
        base_dir = Path(__file__).resolve().parents[2]
        model_path = base_dir / "trainer" / "embeddingClassifier" / "face_recognition_model.joblib"
        model_data = joblib.load(model_path)
        self.recognition_model = model_data['model']  # fetch model 
        self.scaler = model_data['scaler']            # fetch scaler into ram
        print("[INFO] Recognition model loaded")
        self.last_known_name = "unknown" 
        self.last_known_face = None 


    def upload_and_send_metadata(self, face_img, encoding=None, name="default value", camera_id="cam_01"): ## default value

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
                print("[ERROR] No image path returned from server")
                return

            image_path = uploaded_paths[0]

            if encoding is None:
                rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb)
                if not encodings:
                    print("[ERROR] No face encoding found")
                    return
                encoding = encodings[0]

            metadata = {
                "name": name,
                "camera_id": camera_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "embedding": encoding.tolist(),
                "image_path": image_path,
                "confidence": 0.95
            }

            # Send metadata
            meta_res = requests.post("http://localhost:8889/api/InputFaceRecognitionDetails", json=metadata)
            if meta_res.status_code == 200:
                print(f"[INFO] Upload + Metadata OK: {image_path}")
            else:
                print(f"[ERROR] Metadata failed: {meta_res.text}")

        except Exception as e:
            print(f"[ERROR] Exception in upload_and_send_metadata: {e}")

    def recognize_face(self, encoding, threshold=0.6):
        if encoding is None:
            return "unknown", 0.0

        encoding = np.array(encoding).reshape(1, -1)
        try:
            prediction = self.recognition_model.predict(encoding)
            probabilities = self.recognition_model.predict_proba(encoding)

            predicted_name = prediction[0]
            confidence_idx = list(self.recognition_model.classes_).index(predicted_name)
            confidence = probabilities[0][confidence_idx]

            if confidence >= threshold:
                return predicted_name, confidence
            else:
                return "unknown", confidence

        except Exception as e:
            print(f"[ERROR] Recognition failed: {e}")
            return "unknown", 0.0

    def run(self):
        try:
            last_recognized_faces = []
            frame_count = 0
            start_time = time.time()
            fps = 0

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[ERROR] Failed to capture frame")
                    time.sleep(0.1)
                    continue
                frame_count += 1
                if frame_count % 10 == 0:
                    fps = frame_count / (time.time() - start_time)
                    start_time = time.time()
                    frame_count = 0
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                faces = self.detector.detect(frame)
                now = time.time()
                process_now = (now - self.last_checked_time > self.save_interval)

                updated_faces = []

                for (x1, y1, x2, y2, conf) in faces:
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    face_img = frame[y1:y2, x1:x2]

                    if face_img is None or face_img.size == 0 or face_img.shape[0] < 30 or face_img.shape[1] < 30:
                        continue

                    name = "unknown"
                    confidence = 0.0

                    if process_now:
                        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        encodings = face_recognition.face_encodings(rgb)
                        if encodings:
                            current_encoding = encodings[0]

                            if self.mode == "collect":
                                self.upload_and_send_metadata(face_img, encoding=current_encoding, name="collecting")

                            elif self.mode == "recognize":
                                name, confidence = self.recognize_face(current_encoding)
                                if self.face_filter.is_different_with_encoding(current_encoding):
                                    print(f"[INFO] ✅ New face detected at {now:.2f}, sending...")
                                    print(f"detected name: {name}")
                                    self.upload_and_send_metadata(face_img, encoding=current_encoding, name=name)
                            
                            updated_faces.append((x1, y1, x2, y2, name, confidence))
                    else:
                        name = "unknown"
                        confidence = 0.0

                        for (lx1, ly1, lx2, ly2, lname, lconf) in last_recognized_faces:
                            if abs(x1 - lx1) < 30 and abs(y1 - ly1) < 30:
                                name = lname
                                confidence = lconf
                                break
                        updated_faces.append((x1, y1, x2, y2, name, confidence))

                if process_now:
                    last_recognized_faces = updated_faces
                    self.last_checked_time = now

                # Draw
                for (x1, y1, x2, y2, name, confidence) in updated_faces:
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (color), 2)
                    cv2.putText(frame, f"Probability: {confidence:.2f}", (x1, y1-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                    cv2.putText(frame, name, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (color), 2)

                cv2.imshow("Face Detection", frame)
                if cv2.waitKey(1) == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Camera stopped")