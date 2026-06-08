# Isolation Forest Anomaly Detector

A Streamlit application for detecting anomalies in datasets using the Isolation Forest algorithm.

## Features

- **Automatic Model Training**: Models are trained on app startup if they don't exist
- **Upload or Use Sample Data**: Upload your own CSV or use the provided sample dataset
- **Adjustable Contamination**: Tune the expected percentage of anomalies
- **Interactive Results**: View anomaly scores and detailed results
- **Download Results**: Export the dataset with anomaly labels

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model (optional - app auto-trains):
```bash
python train.py
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Dataset Format

The app expects CSV files with numeric columns. Non-numeric columns are automatically ignored.

## How it Works

Isolation Forest is an unsupervised algorithm that isolates anomalies by randomly selecting features and split values. Anomalies are identified as data points that are isolated more easily than normal points.

## Customization

- **Contamination**: Adjust the slider to change the expected percentage of anomalies
- **Model Parameters**: Edit `train.py` to adjust n_estimators, random_state, or other hyperparameters
