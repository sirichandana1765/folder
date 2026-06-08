import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'anydataset.csv')


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f'Dataset not found at {CSV_PATH}.')

    df = pd.read_csv(CSV_PATH)
    X = df.select_dtypes(include=['number']).fillna(0)
    if X.shape[1] < 1:
        raise ValueError('No numeric columns found in the dataset.')

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
    iso_forest.fit(Xs)
    
    with open(os.path.join(MODEL_DIR, 'isolation_forest.pkl'), 'wb') as f:
        pickle.dump(iso_forest, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved Isolation Forest and scaler to {MODEL_DIR}')


if __name__ == '__main__':
    main()
