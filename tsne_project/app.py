import os
import pandas as pd
import streamlit as st
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


MODEL_DIR = os.path.join('tsne_project', 'models')


def main():
    st.title('t-SNE Explorer')
    st.markdown('Upload a CSV downloaded from Kaggle (manually) or use the sample dataset in `tsne_project/data/anydataset.csv`.')

    uploaded = st.file_uploader('Upload CSV', type=['csv'])
    use_sample = st.checkbox('Use sample dataset (tsne_project/data/anydataset.csv)', value=True)

    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f'Failed to read uploaded CSV: {e}')
    elif use_sample:
        sample_path = os.path.join('tsne_project', 'data', 'anydataset.csv')
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
        else:
            st.warning('Sample dataset not found at tsne_project/data/anydataset.csv')

    if df is None:
        st.info('Provide a dataset to compute t-SNE.')
        return

    st.subheader('Dataset (preview)')
    st.dataframe(df.head())

    numeric = df.select_dtypes(include=['number']).fillna(0)
    if numeric.empty:
        st.error('No numeric columns found in the dataset.')
        return

    n_samples = len(numeric)
    
    # t-SNE requires at least 15 samples for reasonable results
    if n_samples < 15:
        st.error(f'❌ Your dataset has only {n_samples} samples. t-SNE requires at least 15 samples to work properly.')
        return
    
    # t-SNE requires perplexity < (n_samples - 1) / 3
    max_perplexity = max(5, (n_samples - 1) // 3)
    default_perplexity = min(30, max_perplexity)
    
    perplexity = st.slider('Perplexity', 5, min(50, max_perplexity), default_perplexity)
    learning_rate = st.slider('Learning rate', 10, 1000, 200)
    n_iter = st.slider('Iterations', 250, 5000, 1000)
    do_pca = st.checkbox('Run initial PCA (recommended for high-dimensional data)', value=True)

    if st.button('Run t-SNE'):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(numeric)
        if do_pca:
            n_pca = min(50, Xs.shape[1])
            pca = PCA(n_components=n_pca)
            Xp = pca.fit_transform(Xs)
        else:
            Xp = Xs

        try:
            tsne = TSNE(n_components=2, perplexity=float(perplexity), learning_rate=float(learning_rate), max_iter=n_iter, init='pca', random_state=42)
            emb = tsne.fit_transform(Xp)
            emb_df = pd.DataFrame(emb, columns=['tsne1', 'tsne2'])
            st.subheader('t-SNE embedding (first rows)')
            st.dataframe(emb_df.head())
            csv = pd.concat([df.reset_index(drop=True), emb_df], axis=1)
            st.download_button('Download dataset with embedding', csv.to_csv(index=False), file_name='tsne_with_embedding.csv')
        except ValueError as e:
            st.error(f't-SNE failed: {str(e)}. Try reducing perplexity or iterations.')


if __name__ == '__main__':
    main()
