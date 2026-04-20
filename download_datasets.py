# download_datasets.py
import os
import zipfile
import urllib.request
from config import STARE_DIR

print("Downloading STARE dataset from mirror...")
stare_url = "https://bj.bcebos.com/paddleseg/dataset/stare/stare.zip"

urllib.request.urlretrieve(stare_url, "stare.zip")
print("Download complete. Extracting...")

with zipfile.ZipFile("stare.zip", 'r') as zip_ref:
    zip_ref.extractall(STARE_DIR)

os.remove("stare.zip")
print(f"STARE dataset extracted to {STARE_DIR}")