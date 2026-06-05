import os
import sys
import requests
import shutil
import subprocess


def fetch_if_missing(dest_path=None, url=None):
    dest_path = dest_path or os.path.join('project', 'data', 'anydataset.csv')
    dest_dir = os.path.dirname(dest_path)
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.exists(dest_path):
        print(f'Data found at {dest_path}')
        return True

    url = url or os.getenv('DATA_URL')
    if not url:
        print('No DATA_URL provided. Set the DATA_URL environment variable to a raw file URL or Kaggle dataset identifier.')
        return False

    # Support Kaggle: DATA_URL like "owner/dataset" or full kaggle URL
    if 'kaggle' in url and not url.startswith('http'):
        try:
            print('Attempting to download from Kaggle using the kaggle CLI...')
            # Expecting dataset identifier like 'username/dataset-name'
            subprocess.check_call(['kaggle', 'datasets', 'download', '-d', url, '-p', dest_dir, '--unzip'])
            # Try to find a CSV file in dest_dir
            for fname in os.listdir(dest_dir):
                if fname.lower().endswith('.csv'):
                    shutil.move(os.path.join(dest_dir, fname), dest_path)
                    return True
        except Exception as e:
            print('Kaggle download failed:', e)
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
