import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import csv

# --- CONFIGURATION ---
# Bounding box for the Great Lakes region (approximate)
# (min_lon, min_lat, max_lon, max_lat)
BBOX = (-93.0, 41.0, -75.0, 50.0)
# Local image file of the Great Lakes region
IMAGE_FILE = "lakes.png"
# Number of points to sample
N_POINTS = 10000
# Output CSV file
OUTPUT_CSV = "great_lakes_dataset.csv"

# --- DOWNLOAD AND LOAD IMAGE ---

# Download image with error handling
# Load the local image file
try:
    img = Image.open(IMAGE_FILE).convert("RGB")
except Exception as e:
    raise RuntimeError(f"Could not open local image '{IMAGE_FILE}'. Error: {e}")
width, height = img.size

# --- HELPER FUNCTIONS ---
def latlon_to_pixel(lon, lat, bbox, img_size):
    min_lon, min_lat, max_lon, max_lat = bbox
    width, height = img_size
    x = int((lon - min_lon) / (max_lon - min_lon) * width)
    y = int((max_lat - lat) / (max_lat - min_lat) * height)
    return x, y

def is_water(pixel):
    # Simple heuristic: water is more blue than green or red
    r, g, b = pixel
    return b > 100 and b > r and b > g

# --- SAMPLE POINTS AND LABEL ---
data = []
for _ in range(N_POINTS):
    lon = np.random.uniform(BBOX[0], BBOX[2])
    lat = np.random.uniform(BBOX[1], BBOX[3])
    x, y = latlon_to_pixel(lon, lat, BBOX, (width, height))
    if 0 <= x < width and 0 <= y < height:
        pixel = img.getpixel((x, y))
        label = 1 if is_water(pixel) else 0  # 1=water, 0=land
        data.append((lon, lat, label))

# --- SAVE TO CSV ---
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["longitude", "latitude", "label"])  # header
    writer.writerows(data)

print(f"Saved {len(data)} points to {OUTPUT_CSV}")
