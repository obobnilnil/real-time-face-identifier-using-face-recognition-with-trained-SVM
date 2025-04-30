import requests
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def fetch_embeddings(name):
    try:
        print(f"[INFO] Fetching data for {name}...")
        response = requests.get(f"http://localhost:8889/api/FetchEmbeddingDetailsForModelTrainer?name={name}", timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("embeddings", [])
    except Exception as e:
        print(f"[ERROR] Fetching embeddings for {name} failed: {e}")
        return []

# Fetch both sets
label_a = "person1"
label_b = "person2"

embeddings_a = fetch_embeddings(label_a)
embeddings_b = fetch_embeddings(label_b)


# Check if both have data
if not embeddings_a or not embeddings_b:
    print("[ERROR] Missing embeddings for one or both classes.")
    exit(1)

# 2. Prepare data
X = []
y = []

for item in embeddings_a:
    try:
        X.append(np.array(item['Embedding']))
        y.append('label_a')
    except KeyError as e:
        print(f"[WARNING] Missing key for {label_a}: {e}")

for item in embeddings_b:
    try:
        X.append(np.array(item['Embedding']))
        y.append('label_b')
    except KeyError as e:
        print(f"[WARNING] Missing key for {label_b}: {e}")

# Check data consistency
if len(X) != len(y):
    print("[ERROR] Mismatched data lengths between X and y")
    exit(1)

X = np.array(X)
y = np.array(y)

# 3. Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 5. Train model
print("[INFO] Training SVM model...")
model = SVC(probability=True, kernel='linear')
model.fit(X_train, y_train)

# 6. Evaluate
print("[INFO] Evaluating model...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Save model
model_data = {
    'model': model,
    'scaler': scaler,
    'metadata': {
        'training_samples': len(X_train),
        'classes': model.classes_.tolist(),
        'api_endpoint': 'GetEmbeddingDetailsForModelTrainer'
    }
}

joblib.dump(model_data, 'face_recognition_model.joblib')
print("[SUCCESS] Model saved as face_recognition_model.joblib")
