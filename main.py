import pandas as pd
import streamlit as st
import numpy as np
from sklearn.utils.validation import joblib

# st.markdown("# 1. Information")
# create content


def main():
    st.title("Halaman Informasi")
    st.header("KLASIFIKASI BERITA")
    st.container()
    st.write("""
Klasifikasi Berita adalah proses pengelompokan atau kategorisasi berita ke dalam kelas-kelas tertentu berdasarkan kesamaan karakteristik, topik, atau tema. Tujuannya adalah untuk memudahkan pencarian, pengorganisasian, dan analisis berita.
            """)

    st.header("Informasi Data")

    # Crowling data
    st.write("""
    Data diperoleh dari hasil crowling data dari website https://pta.trunojoyo.ac.id
    Data yang diambil berasal dari prodi Teknik Informatika, berikut variabel/fitur yang dihasilkan dari proses crowling data yaitu:
    * Judul
    * Penulis
    * Dosen pembimbing I
    * Dosen pembimbing II
    * Abstrak
    """)


if __name__ == "__main__":
    main()
