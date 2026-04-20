from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NASA_CMAPSS_ZIP_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
DEFAULT_DATASETS = ("FD001",)


def download_zip(destination: Path) -> None:
    response = requests.get(NASA_CMAPSS_ZIP_URL, stream=True, timeout=60)
    response.raise_for_status()
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def extract_datasets(zip_path: Path, destination: Path, datasets: tuple[str, ...]) -> None:
    names = {f"train_{dataset}.txt" for dataset in datasets}
    names.update({f"test_{dataset}.txt" for dataset in datasets})
    names.update({f"RUL_{dataset}.txt" for dataset in datasets})
    names.add("readme.txt")

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            target_name = Path(member).name
            if target_name not in names:
                continue
            extracted_path = destination / target_name
            with archive.open(member) as source, extracted_path.open("wb") as target:
                shutil.copyfileobj(source, target)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "CMAPSSData.zip"
    print(f"Downloading NASA C-MAPSS archive from {NASA_CMAPSS_ZIP_URL}")
    download_zip(zip_path)
    extract_datasets(zip_path, DATA_DIR, DEFAULT_DATASETS)
    print("Downloaded and extracted:")
    for dataset in DEFAULT_DATASETS:
        print(f"  - train_{dataset}.txt")
        print(f"  - test_{dataset}.txt")
        print(f"  - RUL_{dataset}.txt")


if __name__ == "__main__":
    main()
