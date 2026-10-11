import os
import zipfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
DATA_ZIP_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
ZIP_PATH = DATA_DIR / "CMAPSSData.zip"

def download_and_extract():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if files already extracted
    expected_files = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
    if all((DATA_DIR / f).exists() for f in expected_files):
        print(f"Dataset already exists in {DATA_DIR}")
        return True

    print(f"Downloading NASA C-MAPSS dataset from {DATA_ZIP_URL}...")
    try:
        urllib.request.urlretrieve(DATA_ZIP_URL, ZIP_PATH)
        print("Download complete. Extracting archive...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
            
        # Clean up any nested directories or zip
        for item in DATA_DIR.iterdir():
            if item.is_file() and item.suffix == ".zip":
                try:
                    with zipfile.ZipFile(item, 'r') as inner_zip:
                        inner_zip.extractall(DATA_DIR)
                except Exception:
                    pass
        print(f"Extraction successful. Available files: {[f.name for f in DATA_DIR.glob('*.txt')]}")
        return True
    except Exception as e:
        print(f"Error downloading/extracting dataset: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = download_and_extract()
    expected = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
    if not (success or all((DATA_DIR / f).exists() for f in expected)):
        sys.exit(1)
