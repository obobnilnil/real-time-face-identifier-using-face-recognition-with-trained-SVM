<p align="center">
  <img src="assets/crop4.gif" width="48%" style="margin-right:1%">
  <img src="assets/crop5.gif" width="48%">
</p>
<H1>🔍Introduction</H1>
<p>
  
This project is a **real-time person identification system** that uses the **face_recognition** library for facial embedding extraction and a custom-trained classifier based on the **SVM algorithm** for identity prediction. It is designed as a **microservice architecture** comprising four main components:

- a **Python-based camera module** for face detection  
- a **Go backend service** for handling API and database operations  
- a **Python model training service**  
- and **MongoDB** for storing facial embeddings and metadata
## ⚙️ Environment
- **Camera module (Python)**: Runs natively
- **Model training (Python)**: Runs natively
- **Backend service (Go)**: Runs natively
- **MongoDB**: Runs inside a Docker container


**A unique feature of this system is** its ability to **not only estimate whether the captured face is real** — based purely on the input image without requiring additional hardware (such as body sensors or temperature detectors) — **but also identify the person by name**, based on a model that is **trained entirely with user-provided data**.

## 🔧 System Components

- **Neural Network Inference**  
  `face_recognition` library using dlib's CNN-based model for face embedding extraction

- **Detection**  
  OpenCV-based face detector using `res10_300x300_ssd` pretrained model

- **Recognition**  
  Custom-trained SVM classifier (`sklearn.SVC`) for name prediction based on embeddings

- **Embedding Storage**  
  MongoDB database to store face embeddings and associated metadata

- **API Backend**  
  Go backend service using Gin to manage API endpoints and communicate with database

- **Training Pipeline**  
  Python script to fetch embeddings from the database and train SVM model with `joblib` export

As shown in the image below, the system can **detect multiple faces in real time**, assign names to **recognized individuals**, and indicate a **prediction probability**. Faces that are not recognized are labeled as **unknown** along with their **confidence scores**.

<div align="center">
  <H1>Performace</H1>
    <img src="https://github.com/user-attachments/assets/f5fc2b47-951a-4ffd-83b0-acdf55875756" alt="outlook" width="500"/>
    <img src="https://github.com/user-attachments/assets/072024c4-4269-404b-b4d2-27b7141af5f0" alt="outlook" width="500"/>
    <p><em>Note:</em> I had to set the recognition threshold to <strong>0.4</strong> for this example. If the threshold were set to 0.6, the model would fail to recognize <strong>suvanee because the training data for her is limited</strong>. In contrast, <strong>worapon has over 50 images</strong> used during training, which leads to more reliable predictions at higher thresholds.</p>
    <img src="https://github.com/user-attachments/assets/4ec64099-1cb0-4378-b365-c07161664b24" alt="outlook" width="500"/>
    <p><em>Note:</em> For the person on the left, the model shows a probability of 0.60 with the label <strong>unknown because the predicted confidence score did not exceed the classification threshold for any known identity</strong>. As a result, the system intentionally classifies the face as <strong>unknown</strong>.</p>
</div>

- The trained model in this example supports two classes: **worapon** and **suvanee**, each representing a person's identity based on facial embeddings.
- **If you want to use this system with your own data, you will need to collect face images and train the model accordingly (the provided code shows how to do this)**.
- The `probability` parameter refers to the **confidence score returned by the SVM classifier**, indicating how likely the input embedding matches the predicted class.

---

### 📌 Note on Training Data

The current model was trained using:

- **50 images of worapon**
- **20 images of suvanee**

If you require a higher **prediction confidence** (e.g., `probability > 0.7–0.8`), it is **strongly recommended** to collect more training samples per identity.  
This will improve model accuracy and reduce the chance of misclassification, especially when working with similar-looking faces or under low-light conditions.


## 🔧 Prerequisites
- Only one MongoDB database is required
- Make sure MongoDB is running before starting the backend service

## 🚀 Start the Face Recognition Pipeline

1. **Clone the repository**

```bash
git clone https://github.com/obobnilnil/real-time-face-identifier-using-face-recognition-with-trained-SVM.git
cd real-time-face-identifier-using-face-recognition-with-trained-SVM
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
```

```bash
# On Windows
venv\Scripts\activate
```

```bash
# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies for both modules**

```bash
cd camera/MLproject
pip install -r requirements.txt

cd ../../trainer/embeddingClassifier
pip install -r requirements.txt
```

4. **Start the backend service**

```bash
cd ../../backend/ml_database
go run main.go
```
After running this command, the MongoDB database will be initialized and ready to store embedding data.

5. **Start the Python-based camera in data collection mode**

To collect training data for a new person, you'll need to run the camera module in `"collect"` mode and specify your desired name.

**Step 5.1**: Edit `ml_project.py`

At line 12, change the mode to `"collect"`:

```python
app = CameraApp(detector, mode="collect")
```

**Step 5.2**: Edit `camera.py`

At line 151 (inside `self.upload_and_send_metadata(...)`), replace the default name `"collecting"` with your actual name. For example:

```python
self.upload_and_send_metadata(face_img, encoding=current_encoding, name="your_name")
```

Replace `"your_name"` with something like `"worapon"` or `"suvanee"` depending on who you're collecting data for.

**Step 5.3**: Run the camera module

```bash
cd ../../camera/MLproject
python ml_project.py
```
This will activate your webcam and start collecting facial data under the given name. Pictures are taken every 2 seconds.
Although this may seem complicated, using a single camera makes this method more convenient for me.

6. **Train the model using the SVM classifier**

After you have completed collecting face data using the camera module, you can train the model using the provided script.

Run the training script

```bash
cd ../../trainer/embeddingClassifier
python embeddingClassifier.py
```

After running this command, a new file called `face_recognition_model.joblib` will be created. This file contains the trained classifier and will be used later for real-time face recognition.

7. **Start the system in recognition mode**

After training the model, you can now run the camera module in `"recognize"` mode to perform real-time face identification.

**Step 7.1**: Edit `ml_project.py`

At line 12, switch the mode back to `"recognize"`:

```python
app = CameraApp(detector, mode="recognize")
```

This tells the system to use the trained model (`face_recognition_model.joblib`) to identify known faces.

**Step 7.2**: Run the recognition script

```bash
cd ../../camera/MLproject
python ml_project.py
```
When a face is detected, the system will display the predicted name and confidence score on the screen. Faces that do not match any trained identity above the confidence threshold will be labeled as `unknown`.

## Face Recognition Pipeline Flow
<div align="center">
  <img src="https://github.com/user-attachments/assets/cf1efce5-5c3a-4751-a461-2ae33bdb5890" alt="flow image" width="900"/>
</div>

## Adjustable Parameters

1. **MONGO_DB** – (file: backend/ml_database/.env) 
   Name of the MongoDB database used for storing embeddings and metadata.

2. **label_a, label_b, ...** – (file: trainer/embeddingClassifier/embeddingClassifier.py, lines 21–22)  
   Labels for each identity during model training. You can rename these to match the collected names.

3. **mode="recognize"** – (file: camera/MLproject/ml_project.py, line ~12)  
   Mode for the camera system. Change to `"collect"` when collecting new training data.

4. **save_interval=2** – (file: camera/MLproject/function/logic/faceCam/camera.py, __init__ method)  
   Number of seconds to wait before capturing the next face frame. Controls how frequently images are processed and uploaded.

5. **threshold (in FaceFilter) – default: 0.6**  
   (file: camera/MLproject/function/logic/faceCam/camera.py, line: self.face_filter = FaceFilter(threshold=0.6)) 
   Determines how different two face embeddings must be before the system considers it a new face and sends metadata again. Prevents redundant uploads when the same person stays still.

6. **threshold (in recognize_face) – default: 0.6**  
   (file: camera/MLproject/function/logic/faceCam/camera.py, method: recognize_face()) 
   Minimum confidence score required for the model to assign a name. If the score is below this threshold, the prediction is considered uncertain and the name is set to `unknown`.

7. **name="collecting"** – (file: camera/MLproject/function/logic/faceCam/camera.py, line ~151)
   The label assigned when uploading embeddings in `"collect"` mode. You should change `"collecting"` to your actual identity label, such as `"worapon"` or `"suvanee"`.
   
9. **cam_index=1** – (file: camera/MLproject/function/logic/faceCam/camera.py, __init__ method)  
   Specifies which camera device to use.  
   - 0 is usually the **built-in webcam**  
   - 1 or 2 may be used for **external USB cameras**

10. **Snapshot output folder: snapshots/** – (defined in backend Go service)* 
   When a face is detected and uploaded, the image is saved to the folder `snapshots/` inside the project root. This is handled by the backend service (`UploadImageAndReturnPath`).

## 📡 API Endpoints

The following API endpoints are provided by the Go backend service for interacting with face data, images, and embeddings.

| Method | Endpoint                                      | Description                                                                 |
|--------|-----------------------------------------------|-----------------------------------------------------------------------------|
| POST   | `/api/InputFaceRecognitionDetails`            | Submits facial embedding and metadata (name, timestamp, etc.) for storage. |
| POST   | `/api/UploadImageAndReturnPath`               | Uploads an image file and returns the server-side path.                    |
| GET    | `/api/GetImagesByCamNumber`                   | Retrieves image paths by camera ID and time range.                         |
| GET    | `/api/GetImagesByName`                        | Retrieves image paths by name and optional timestamp.                      |
| GET    | `/api/FetchEmbeddingDetailsForModelTrainer`   | Fetches all stored embeddings for a given name, used for model training.   |

1. POST /api/InputFaceRecognitionDetails
Type: application/json

Body example:
```bash
{
  "name": "worapon",
  "camera_id": "cam_01",
  "timestamp": "2025-04-26T08:13:23Z",
  "embedding": [0.123, 0.456, 0.789, ...],
  "image_path": "snapshots/face_cam01_1714470882937.jpg",
  "confidence": 0.95
}
```
2. POST /api/UploadImageAndReturnPath
Type: multipart/form-data
```bash
camera_id = cam_01
images = face_1745530718.jpg
```
3. GET /api/GetImagesByCamNumber
Query Parameters:
```bash
cameraID=cam_01
timestamp=2025-04-25T07:21:13Z
http://localhost:8889/api/GetImagesByCamNumber?cameraID=cam_01&timestamp=2025-04-25T07:21:13Z
```
4. GET /api/GetImagesByName
Query Parameters:
```bash
name=bank_worapon
timestamp=All (or use a specific ISO string)
http://localhost:8889/api/GetImagesByName?name=bank_worapon&timestamp=All
```
 5. GET /api/FetchEmbeddingDetailsForModelTrainer
Query Parameters:
```bash
name=your_name
http://localhost:8889/api/FetchEmbeddingDetailsForModelTrainer?name=your_name
```
---
Thank you for your interest in this repository.  
If you find this project helpful, feel free to star ⭐ or contribute!

Have a great day!
