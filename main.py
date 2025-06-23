import pandas as pd
import streamlit as st
import numpy as np
from sklearn.utils.validation import joblib

# st.markdown("# 1. Information")
# create content


def main():
    st.title("Halaman Informasi")
    st.header("Topic Modelling")
    st.container()
    st.write("""
            * Topic Modelling adalah suatu metode pada analisa penambangan teks untuk melakukan penemuan data-data teks yang tersembunyi dan menemukan hubungan antara teks yang satu dengan lainnya dari suatu corpus
            * mengelompokkan data teks yang didasarkan pada topik tertentu
            * Ide-tema-pokok bahasan utama dalam suatu kumpulan teks
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
