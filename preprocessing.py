import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google.colab import drive  # For Google Colab
import gdown  # For downloading from Google Drive

def load_data_from_drive():
    """Load data from Google Drive"""
    # Mount Google Drive (for Colab)
    drive.mount('/content/drive')
    
    # Path to your CSV file in Drive
    drive_path = '/content/drive/MyDrive/streamlit/df.csv'  # Update with your path
    
    # If running locally with gdown:
    # file_id = "YOUR_GOOGLE_DRIVE_FILE_ID"  # Get from shareable link
    # url = f"https://drive.google.com/uc?id={file_id}"
    # output = "df.csv"
    # gdown.download(url, output, quiet=False)
    
    if os.path.exists(drive_path):
        return pd.read_csv(drive_path, encoding='utf-8')
    else:
        st.error("File not found in Google Drive path!")
        return None

def preprocessing_page():
    st.title("📊 Preprocessing Data Teks")
    
    # Load data from Drive instead of local file
    df = load_data_from_drive()
    if df is None:
        return
    
    # Rest of your preprocessing code remains the same...
    kategori_mapping = {'Travel': 0, 'Edukasi': 1, 'Sports': 2, 'Politik': 3, 'Health': 4}
    reverse_mapping = {v: k for k, v in kategori_mapping.items()}

    if 'Label' not in df.columns and 'Kategori' in df.columns:
        df['Label'] = df['Kategori'].replace(kategori_mapping)

    st.subheader("1. Data Awal")
    st.dataframe(df.head())

    # [Rest of your existing preprocessing code...]

if __name__ == "__main__":
    preprocessing_page()
