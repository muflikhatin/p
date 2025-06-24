import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gdown  # For downloading from Google Drive

def load_data():
    """Load data from either local file or Google Drive"""
    # First try local file
    if os.path.exists("df.csv"):
        return pd.read_csv("df.csv", encoding='utf-8')
    
    # If not found locally, try downloading from Google Drive
    try:
        # Replace with your Google Drive file ID (from shareable link)
        file_id = "YOUR_GOOGLE_DRIVE_FILE_ID"  
        url = f"https://drive.google.com/uc?id={file_id}"
        output = "df.csv"
        gdown.download(url, output, quiet=False)
        return pd.read_csv(output, encoding='utf-8')
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return None

def preprocessing_page():
    st.title("📊 Preprocessing Data Teks")
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Could not load data file. Please ensure df.csv exists locally or provide a valid Google Drive link.")
        return
    
    # Category mapping
    kategori_mapping = {'Travel': 0, 'Edukasi': 1, 'Sports': 2, 'Politik': 3, 'Health': 4}
    reverse_mapping = {v: k for k, v in kategori_mapping.items()}

    if 'Label' not in df.columns and 'Kategori' in df.columns:
        df['Label'] = df['Kategori'].replace(kategori_mapping)

    st.subheader("1. Data Awal")
    st.dataframe(df.head())

    # Rest of your preprocessing code...
    if 'Label' in df.columns:
        st.subheader("2. Distribusi Kategori")
        label_counts = df['Label'].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(label_counts.index, label_counts.values, color='skyblue')
        ax.set_title("Distribusi Kategori")
        ax.set_xlabel("Kategori")
        ax.set_ylabel("Jumlah")
        ax.grid(axis='y')
        ax.set_xticks(list(label_counts.index))
        ax.set_xticklabels([reverse_mapping[i] for i in label_counts.index], rotation=45)

        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    int(bar.get_height()), ha='center', va='bottom')

        st.pyplot(fig)

        st.markdown("**Keterangan Label Kategori:**")
        for kategori, kode in kategori_mapping.items():
            st.markdown(f"- **{kode}** = {kategori}")

    # Continue with the rest of your preprocessing code...

if __name__ == "__main__":
    preprocessing_page()
