"""Download Penn Action dataset"""

import subprocess
import tarfile
from pathlib import Path
from .config import RAW_DATA_DIR, DATASET_URL


def download_dataset():
    """Download Penn Action dataset from UPenn server"""

    tar_path = RAW_DATA_DIR / "Penn_Action.tar.gz"

    if tar_path.exists():
        print(f"Dataset already exists at {tar_path}")
        return tar_path

    print(f"Downloading Penn Action dataset from {DATASET_URL}...")
    print("This may take a few minutes (3GB file)...")

    # Use wget to download
    result = subprocess.run(
        ["wget", "-O", str(tar_path), DATASET_URL], capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Download failed: {result.stderr}")

    print(f"Downloaded to {tar_path}")
    return tar_path


def extract_dataset(tar_path):
    """Extract the downloaded tar.gz file"""

    extract_path = RAW_DATA_DIR

    if (extract_path / "labels").exists():
        print(f"Dataset already extracted at {extract_path}")
        return extract_path

    print(f"Extracting {tar_path}...")

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_path)

    print(f"Extracted to {extract_path}")
    return extract_path


if __name__ == "__main__":
    tar_path = download_dataset()
    extract_dataset(tar_path)
