import pandas as pd
import streamlit as st
import numpy as np
import joblib

def main():
    st.title("Halaman Informasi")
    st.header("Klasifikasi Berita")
    st.container()
    st.write("""
            * Text Classification adalah metode dalam analisis teks yang digunakan untuk memberikan label atau kategori tertentu pada data teks berdasarkan isi atau makna dari teks tersebut.
            * Mengelompokkan data teks ke dalam kelas-kelas yang telah ditentukan sebelumnya secara otomatis.
            * Fokus pada identifikasi kategori utama atau label yang relevan dari suatu kumpulan teks.
            """)

    st.header("Informasi Data")
    st.write("""
    Data diperoleh dari hasil crowling data dari portal berita https://www.kompas.com
    Data yang diambil berasal dari website kompas,com, berikut variabel/fitur yang dihasilkan dari proses crowling data yaitu:
    * Kategori
    * Judul
    * Tanggal
    * Jumlah_Kata
    * Link
    """)

if __name__ == "__main__":
    main()