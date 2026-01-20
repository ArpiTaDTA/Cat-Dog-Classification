import cv2
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

IMG_SIZE = 64
data = []
labels = []

# Load images
for category in ["cats", "dogs"]:
    label = 0 if category == "cats" else 1
    path = os.path.join("dataset", category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image.flatten()
        data.append(image)
        labels.append(label)

X = np.array(data)
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train models
svm = SVC()
rf = RandomForestClassifier()
lr = LogisticRegression(max_iter=1000)
kmeans = KMeans(n_clusters=2)

svm.fit(X_train, y_train)
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)
kmeans.fit(X_train)

# Save models
os.makedirs("models", exist_ok=True)

joblib.dump(svm, "models/svm.pkl")
joblib.dump(rf, "models/rf.pkl")
joblib.dump(lr, "models/lr.pkl")
joblib.dump(kmeans, "models/kmeans.pkl")

print("✅ All models trained and saved successfully!")
