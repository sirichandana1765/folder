import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

from fetch_data import fetch_if_missing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'anydataset.csv')


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    # attempt to fetch dataset if missing; set DATA_URL env var if you want to download from GitHub or Kaggle
    fetch_if_missing(dest_path=CSV_PATH)
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f'Dataset not found at {CSV_PATH}. Set DATA_URL or place file there.')

    df = pd.read_csv(CSV_PATH)
    X = df.select_dtypes(include=['number']).fillna(0)
    if X.shape[1] < 1:
        raise ValueError('No numeric columns found in the dataset.')

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_components = min(2, Xs.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(Xs)
    with open(os.path.join(MODEL_DIR, 'pca_model.pkl'), 'wb') as f:
        pickle.dump(pca, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved PCA and scaler to {MODEL_DIR}')


if __name__ == '__main__':
    main()
