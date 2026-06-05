# t-SNE Project

This folder contains a simple t-SNE demo. By design the project does not auto-download datasets from Kaggle — please download the CSV manually from Kaggle and either:

- place the CSV at `tsne_project/data/anydataset.csv`, or
- upload the CSV using the Streamlit UI below.

Run training (if you placed the CSV in `tsne_project/data/`):

```bash
pip install -r tsne_project/requirements.txt
python tsne_project/train_tsne.py
```

Or run the interactive Streamlit app and upload the dataset there:

```bash
pip install -r tsne_project/requirements.txt
streamlit run tsne_project/app.py
```
