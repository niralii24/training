import os
import pandas as pd
import requests
from tqdm import tqdm

EXCEL_FILE = "dataset.xlsx" 
OUTPUT_DIR = "downloaded_audio"
TIMEOUT = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_excel(EXCEL_FILE)

required_cols = [ "audio","audio_id"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

print(f"Total files to download: {len(df)}\n")

for _, row in tqdm(df.iterrows(), total=len(df)):


    url = row["audio"]
    audio_id = row["audio_id"]

    save_folder = os.path.join(OUTPUT_DIR)
    os.makedirs(save_folder, exist_ok=True)

    file_path = os.path.join(save_folder, f"{audio_id}.mp3")

    if os.path.exists(file_path):
        continue

    try:
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            f.write(response.content)

    except Exception as e:
        print(f"Failed to download {dp_id}: {e}")

print("\nDownload completed.")