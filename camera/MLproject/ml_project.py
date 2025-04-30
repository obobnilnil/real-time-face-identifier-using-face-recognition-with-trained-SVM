from function.logic.faceCam.detector import FaceDetector
from function.logic.faceCam.camera import CameraApp
from data.deploy import model, config

if __name__ == "__main__":
    model = "data/res10_300x300_ssd_iter_140000.caffemodel"
    config = "data/deploy.prototxt"
    detector = FaceDetector(model, config)
    # app = CameraApp(detector)
    # - "collect": collect face data for training a model
    # - "recognize": use a trained model to recognize faces
    app = CameraApp(detector, mode="recognize")
    app.run()
