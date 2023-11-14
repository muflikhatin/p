from nltk.corpus import stopwords
import pandas as pd
import streamlit as st
import numpy as np
import requests
import nltk
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
import re
import string


def main():
    st.title("Tugas 2 Normalisasi text")
    st.write("""
            Data preprocessing adalah proses persiapan dan perbaikan data sebelum digunakan dalam analisis atau pemodelan. 
            Tujuan utama dari data preprocessing adalah untuk menghilangkan noise, mengatasi masalah data yang tidak lengkap atau tidak konsisten, dan mengubah data menjadi format yang lebih sesuai untuk analisis atau pemodelan yang akan datang. 
            Punctuation process menghilangkan tanda baca dan simbol

            - Stopword

            - Tokenisasi

            - Steeming

            - Feature extraction dan membentuk VSM dalam term frequency, logarithm freqency, one-hot encoding, TF-IDF
        """)

    abst = pd.read_csv("hasil_crowling.csv")
    # st.header("dataset hasil crawling")
    # st.dataframe(abst)

    st.header("Abstrak Sebelum di Preprocessing")
    st.dataframe(abst['Abstrak'])

    nltk.download('stopwords')
    nltk.download('punkt')

    def download_custom_stopwords(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                stopwords_text = file.read()
            custom_stopwords = set(stopwords_text.splitlines())
            return custom_stopwords
        except FileNotFoundError as e:
            print(f"File daftar kata-kata stop words tidak ditemukan: {e}")
            return set()

    # Membaca daftar kata-kata stop words khusus dari file 'daftar_stopword.txt'
    custom_stopwords = download_custom_stopwords('D:/semester 7/streamlit/daftar_stopword.txt')
        # 'C:/Users/user/Documents/ppw/streamlit/daftar_stopword.txt'

    # Import stopwords dalam bahasa Indonesia
    stop_words = set(stopwords.words('indonesian'))

    # Gabungkan stopwords bawaan dengan custom_stopwords
    stop_words.update(custom_stopwords)

    def preprocess_text(teks):
        if pd.isna(teks):
            teks = ""
        else:
            # Menghapus angka
            num_clean = re.sub(r"\d+", "", teks)

            # Punctuation process (Hapus tanda baca)
            teks_clean = num_clean.translate(
                str.maketrans('', '', string.punctuation))

            # Tokenisasi
            tokens = nltk.word_tokenize(teks_clean)

            # Stopwords removal
            teks_cleaned = [
                word for word in tokens if word.lower() not in stop_words]

            # Stemming
            stemmer = PorterStemmer()
            stemmed_tokens = [stemmer.stem(word) for word in teks_cleaned]

            # Gabung kembali tokens menjadi teks
            teks = ' '.join(stemmed_tokens)

        return teks

    # Menambahkan kolom untuk setiap tahap preprocessing
    abst['cleaning'] = abst['Abstrak'].apply(preprocess_text)
    abst['tokenized'] = abst['cleaning'].apply(
        nltk.word_tokenize)  # Tokenizing
    abst['stopwords_removed'] = abst['tokenized'].apply(
        lambda x: [word for word in x if word.lower() not in stop_words])  # Stopword removal
    abst['stemmed'] = abst['stopwords_removed'].apply(
        lambda x: [PorterStemmer().stem(word) for word in x])

    st.header("Hasil Tokenizing")
    # display_text_data(abst['tokenized'], 'Tokenized Text')
    st.dataframe(abst['tokenized'])

    st.header("Hasil Stopwords Removed")
    # display_text_data(abst['stopwords_removed'],
    #                   'Text after Stopwords Removal')
    st.dataframe(abst['stopwords_removed'])

    st.header("Hasil Stemming")
    # display_text_data(abst['stemmed'], 'Text after Stemming')
    st.dataframe(abst['stemmed'])

    st.header("Hasil Preprocessing Data")
    # display_text_data(
    #     abst['cleaning'], 'Text after Cleaning (Punctuation, Numbers, Stopwords, Stemming)')
    st.dataframe(abst['cleaning'])

    abst.to_excel('HasilPreposPTA.xlsx', index=False)


if __name__ == "__main__":
    main()
