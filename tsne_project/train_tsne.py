import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib

from fetch_data import fetch_if_missing


def main():
    os.makedirs(os.path.join('tsne_project', 'models'), exist_ok=True)
    # Expect the dataset to be provided manually (place CSV in tsne_project/data/ or upload via the Streamlit UI)
    csv_path = os.path.join('tsne_project', 'data', 'anydataset.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Dataset not found at {csv_path}. Please download from Kaggle and place the CSV there, or use the Streamlit app to upload it.')

    df = pd.read_csv(csv_path)
    X = df.select_dtypes(include=['number']).dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # optional initial PCA to speed up t-SNE on higher-dimensional data
    n_pca = min(50, Xs.shape[1])
    pca = PCA(n_components=n_pca)
    Xp = pca.fit_transform(Xs)

    tsne = TSNE(n_components=2, init='pca', random_state=42)
    emb = tsne.fit_transform(Xp)

    joblib.dump({'embedding': emb, 'index': df.index.tolist()}, os.path.join('tsne_project', 'models', 'tsne_embedding.pkl'))
    joblib.dump(scaler, os.path.join('tsne_project', 'models', 'scaler.pkl'))
    joblib.dump(pca, os.path.join('tsne_project', 'models', 'pca.pkl'))
    print('Saved t-SNE embedding and helpers to tsne_project/models/')


if __name__ == '__main__':
    main()
