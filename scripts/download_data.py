from pathlib import Path
from urllib.request import urlopen

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "titanic.csv"

def download_dataset(force: bool = False) -> Path:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and not force:
        return DATA_PATH
    try:
        with urlopen(DATA_URL, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"Dataset download returned HTTP {response.status}")
            payload = response.read()
    except Exception as exc:
        raise RuntimeError(
            "Could not download Titanic data. Check your internet connection or download "
            f"the CSV manually from {DATA_URL} into {DATA_PATH}."
        ) from exc
    if not payload.startswith(b"PassengerId,Survived"):
        raise ValueError("Downloaded file does not look like the expected Titanic CSV.")
    DATA_PATH.write_bytes(payload)
    return DATA_PATH

if __name__ == "__main__":
    print(f"Dataset saved to: {download_dataset()}")
