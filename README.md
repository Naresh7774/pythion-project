# AI Fake News Detection System

An intelligent web-based application that automatically detects whether a news article is Fake or Real using Natural Language Processing (NLP) and Machine Learning classification models.

## Project Structure
- `data_downloader.py`: Downloads and standardizes a public fake news dataset.
- `train_model.py`: Performs text preprocessing, trains three different models, picks the best one, and saves it.
- `predict.py`: Contains the `FakeNewsPredictor` class representing the inference pipeline.
- `app.py`: Streamlit front-end application.
- `requirements.txt`: Required dependencies.

## Steps to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset**
   ```bash
   python data_downloader.py
   ```
   *Note: This downloads a generic dataset to guarantee functionality. You can replace `data/dataset.csv` with a Kaggle dataset if it has `title`, `text`, and `label` (Fake/Real) columns.*

3. **Train Machine Learning Model**
   ```bash
   python train_model.py
   ```
   *This step might take a minute or two depending on your CPU, as it fits TF-IDF and multiple models.*

4. **Launch Application**
   ```bash
   streamlit run app.py
   ```
   *Open the Local URL provided in your terminal in a web browser.*
