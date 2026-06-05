import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

from fetch_data import fetch_if_missing


def main():
    os.makedirs(os.path.join('project', 'models'), exist_ok=True)
    # attempt to fetch dataset if missing; set DATA_URL env var if you want to download from GitHub or Kaggle
    fetch_if_missing()
    csv_path = os.path.join('project', 'data', 'anydataset.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Dataset not found at {csv_path}. Set DATA_URL or place file there.')

    df = pd.read_csv(csv_path)
    X = df.select_dtypes(include=['number']).dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    pca.fit(Xs)
    with open(os.path.join('project', 'models', 'pca_model.pkl'), 'wb') as f:
        pickle.dump(pca, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join('project', 'models', 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved PCA and scaler to project/models/')


if __name__ == '__main__':
    main()
