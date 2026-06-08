import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
PCA_PATH = os.path.join(MODEL_DIR, 'pca_model.pkl')
SAMPLE_PATH = os.path.join(BASE_DIR, 'data', 'anydataset.csv')


def train_models():
    """Automatically train models if they don't exist"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(SAMPLE_PATH):
        raise FileNotFoundError(f'Dataset not found at {SAMPLE_PATH}')
    
    df = pd.read_csv(SAMPLE_PATH)
    X = df.select_dtypes(include=['number']).fillna(0)
    
    if X.shape[1] < 1:
        raise ValueError('No numeric columns found in the dataset.')
    
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_components = min(2, Xs.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(Xs)
    
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PCA_PATH, 'wb') as f:
        pickle.dump(pca, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return scaler, pca


@st.cache_resource
def load_models():
    if not os.path.exists(SCALER_PATH) or not os.path.exists(PCA_PATH):
        with st.spinner('Training models... This may take a moment.'):
            train_models()
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(PCA_PATH, 'rb') as f:
        pca = pickle.load(f)
    return scaler, pca


def main():
    st.title('PCA Transform')
    scaler, pca = load_models()

    st.markdown('Upload a CSV or use the sample dataset shipped with the project.')
    uploaded = st.file_uploader('Upload CSV', type=['csv'])
    use_sample = st.checkbox('Use sample dataset (project/data/anydataset.csv)', value=True)

    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f'Failed to read uploaded CSV: {e}')
    elif use_sample:
        if os.path.exists(SAMPLE_PATH):
            df = pd.read_csv(SAMPLE_PATH)
        else:
            st.warning(f'Sample dataset not found at {SAMPLE_PATH}')

    if df is None:
        st.info('Provide a dataset to see PCA components.')
        return

    st.subheader('Dataset (first rows)')
    st.dataframe(df.head())

    numeric = df.select_dtypes(include=['number']).fillna(0)
    if numeric.empty:
        st.error('No numeric columns found in the dataset.')
        return

    n_components = st.slider('Number of PCA components', min_value=1, max_value=min(5, numeric.shape[1]), value=2)
    if st.button('Transform'):
        Xs = scaler.transform(numeric)
        comps = pca.transform(Xs)[:, :n_components]
        st.subheader('PCA Components')
        comps_df = pd.DataFrame(comps, columns=[f'PC{i+1}' for i in range(comps.shape[1])])
        st.dataframe(comps_df.head())
        st.download_button('Download components CSV', comps_df.to_csv(index=False), file_name='components.csv')
        st.write('Explained variance ratio:', list(pca.explained_variance_ratio_[:n_components]))


if __name__ == '__main__':
    main()
