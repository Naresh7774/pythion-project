import os
import pandas as pd
from datasets import load_dataset

def download_dataset():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, "dataset.csv")
    
    print("Downloading a genuine Fake News dataset from HuggingFace (GonzaloA/fake_news)...")
    try:
        # Load the dataset from HuggingFace
        ds = load_dataset('GonzaloA/fake_news')
        
        # Convert the 'train' split to a pandas DataFrame
        df = ds['train'].to_pandas()
        
        # The dataset has a 'label' column where 0 = Fake and 1 = True.
        # Let's map it to match our existing labels ('Fake' and 'Real')
        df['label'] = df['label'].map({0: 'Fake', 1: 'Real'})
        
        # Shuffle the dataset to ensure a good mix of Real and Fake news
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Since the dataset is quite large (~40k rows), let's save a subset (e.g., 5000 rows)
        # to ensure that training is fast but still has a vast real-world vocabulary.
        df_subset = df.head(5000)
        
        df_subset.to_csv(file_path, index=False)
        print(f"\nDataset successfully downloaded and saved to {file_path}")
        print(f"Total authentic and fake records saved: {len(df_subset)}")
    except Exception as e:
        print(f"[ERROR] Failed to download from HuggingFace: {e}")
        print("Please ensure you have an active internet connection.")

if __name__ == "__main__":
    download_dataset()
