import os
import sys
import requests
import shutil
import subprocess


def fetch_if_missing(dest_path=None, url=None):
    dest_path = dest_path or os.path.join('tsne_project', 'data', 'anydataset.csv')
    dest_dir = os.path.dirname(dest_path)
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.exists(dest_path):
        print(f'Data found at {dest_path}')
        return True

    url = url or os.getenv('DATA_URL')
    if not url:
        print('No DATA_URL provided. Set the DATA_URL environment variable to a raw file URL or Kaggle dataset identifier.')
        return False

    # We no longer auto-download from Kaggle. Please download the dataset manually
    # from Kaggle and place the CSV into the target path (or use the Streamlit UI to upload).
    if not url.startswith('http'):
        print('Automatic Kaggle downloads are disabled. Please download the dataset manually from Kaggle and place it at', dest_path)
        return False

    # Generic HTTP/HTTPS download
    if url.startswith('http'):
        try:
            print(f'Downloading data from {url}...')
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print('Saved dataset to', dest_path)
            return True
        except Exception as e:
            print('Download failed:', e)
            return False

    print('Unsupported DATA_URL format')
    return False


if __name__ == '__main__':
    ok = fetch_if_missing()
    sys.exit(0 if ok else 2)
