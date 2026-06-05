import os
import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = os.path.join('project', 'models')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
PCA_PATH = os.path.join(MODEL_DIR, 'pca_model.pkl')


@st.cache_resource
def load_models():
    if not os.path.exists(SCALER_PATH) or not os.path.exists(PCA_PATH):
        return None, None
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    return scaler, pca


def main():
    st.title('PCA Transform')
    scaler, pca = load_models()
    if scaler is None or pca is None:
        st.error('Models not found. Run `project/train.py` first to generate `pca_model.pkl` and `scaler.pkl` in project/models/.')
        st.stop()

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
        sample_path = os.path.join('project', 'data', 'anydataset.csv')
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
        else:
            st.warning('Sample dataset not found at project/data/anydataset.csv')

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
