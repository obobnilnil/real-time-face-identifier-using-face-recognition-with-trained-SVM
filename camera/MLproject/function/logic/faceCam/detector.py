import cv2

class FaceDetector:
    def __init__(self, model_path, config_path, threshold=0.5):
        self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        self.threshold = threshold

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.threshold:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x1, y1, x2, y2) = box.astype(int)
                faces.append((x1, y1, x2, y2, confidence))
        return faces