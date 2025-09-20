import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import joblib
from PIL import Image

# --- CONFIGURATION ---
DATA_CSV = "great_lakes_dataset.csv"
MODEL_FILE = "knn_great_lakes.joblib"
MAP_IMAGE = "lakes.png"
BBOX = (-93.0, 41.0, -75.0, 50.0)  # (min_lon, min_lat, max_lon, max_lat)


# --- LOAD DATA ---
df = pd.read_csv(DATA_CSV)
X = df[["longitude", "latitude"]].values
y = df["label"].values


# --- TRAIN KNN ---
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X, y)


# --- SAVE MODEL ---
joblib.dump(clf, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

width, height = 600, 450  # Set output image size (smaller for easier viewing)
xx, yy = np.meshgrid(
    np.linspace(BBOX[0], BBOX[2], width),
    np.linspace(BBOX[3], BBOX[1], height)  # y axis reversed for image
)

# Ensure Z is integer (0=land, 1=water)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.astype(int).reshape(xx.shape)

plt.figure(figsize=(7.5, 5.6))
# Custom colors: 0=land (light tan), 1=water (light blue)
plt.contourf(xx, yy, Z, alpha=1.0, levels=[-0.5, 0.5, 1.5], colors=["#f5e6c8", "#a6d8ff"])
plt.xticks([])
plt.yticks([])
plt.box(False)
plt.tight_layout(pad=0)
plt.savefig('knn_boundary_only.png', dpi=200, bbox_inches='tight', pad_inches=0)
plt.close()
print("Decision boundary plot saved as knn_boundary_only.png")
