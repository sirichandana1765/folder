import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'isolation_forest.pkl')
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
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
    iso_forest.fit(Xs)
    
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(iso_forest, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return scaler, iso_forest


@st.cache_resource
def load_models():
    if not os.path.exists(SCALER_PATH) or not os.path.exists(MODEL_PATH):
        with st.spinner('Training models... This may take a moment.'):
            train_models()
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(MODEL_PATH, 'rb') as f:
        iso_forest = pickle.load(f)
    return scaler, iso_forest


def main():
    st.title('Isolation Forest Anomaly Detector')
    st.markdown('Detect anomalies in your data using Isolation Forest algorithm.')
    
    scaler, iso_forest = load_models()
    
    uploaded = st.file_uploader('Upload CSV', type=['csv'])
    use_sample = st.checkbox('Use sample dataset (isolation/data/anydataset.csv)', value=True)
    
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
        st.info('Provide a dataset to detect anomalies.')
        return
    
    st.subheader('Dataset (preview)')
    st.dataframe(df.head())
    
    numeric = df.select_dtypes(include=['number']).fillna(0)
    if numeric.empty:
        st.error('No numeric columns found in the dataset.')
        return
    
    contamination = st.slider('Contamination (% of anomalies)', 1, 50, 10) / 100.0
    
    if st.button('Detect Anomalies'):
        with st.spinner('Running Isolation Forest...'):
            X_scaled = scaler.transform(numeric)
            
            # Retrain with selected contamination
            iso_forest_temp = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
            predictions = iso_forest_temp.fit_predict(X_scaled)
            anomaly_scores = iso_forest_temp.score_samples(X_scaled)
            
            # Add results to dataframe
            result_df = df.copy()
            result_df['anomaly'] = predictions
            result_df['anomaly_score'] = anomaly_scores
            result_df['is_anomaly'] = result_df['anomaly'] == -1
            
            # Display results
            st.subheader('Anomaly Detection Results')
            n_anomalies = (result_df['is_anomaly']).sum()
            st.metric('Anomalies Found', n_anomalies, f'{(n_anomalies/len(result_df)*100):.1f}%')
            
            st.subheader('Dataset with Anomaly Labels')
            st.dataframe(result_df)
            
            st.subheader('Anomaly Details')
            anomalies = result_df[result_df['is_anomaly']].sort_values('anomaly_score')
            if len(anomalies) > 0:
                st.write(f"Found {len(anomalies)} anomalies:")
                st.dataframe(anomalies)
            else:
                st.info("No anomalies detected with current settings.")
            
            # Download button
            csv = result_df.to_csv(index=False)
            st.download_button(
                'Download results with anomaly labels',
                csv,
                file_name='anomalies_detected.csv',
                mime='text/csv'
            )


if __name__ == '__main__':
    main()

