# folder

Project PCA demo scaffold.

Structure:

- `project/data/anydataset.csv` — dataset (can be downloaded)
- `project/models/` — saved `pca_model.pkl` and `scaler.pkl`
- `project/train.py` — training script (will attempt to download dataset if missing)
- `project/fetch_data.py` — helper to download a dataset from `DATA_URL` (supports raw HTTP/HTTPS or Kaggle via the `kaggle` CLI)
- `project/notebooks/eda.ipynb` — simple EDA notebook


To use with a dataset hosted on GitHub or any raw URL, set the `DATA_URL` environment variable to the raw file URL, then run:

```bash
export DATA_URL="https://raw.githubusercontent.com/username/repo/branch/path/to/data.csv"
python project/train.py
```

For Kaggle datasets, install the `kaggle` CLI and authenticate, then set `DATA_URL` to the dataset identifier (for example `zynicide/wine-reviews`) and run the same command. The fetcher will try to use the `kaggle` CLI to download and unzip the dataset.

Run the Streamlit UI to interactively transform data with the trained PCA model:

```bash
pip install -r requirements.txt
streamlit run app.py
```

