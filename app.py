# app.py
import streamlit as st
from main import main as main_page
from crowling_data import main as crawling_data_page
from preprocessing_data import main as preprocessing_data_page
from term_frequency import main as term_frequency_page
from K_Means import main as K_Means_page
from topic_modelling import main as topic_modelling_page
from lda_knn_kmeans import main as lda_knn_kmeans_page


# Sidebar untuk navigasi
selected_page = st.sidebar.selectbox(
    "Pilih Halaman:",
    ["Halaman Utama", "Halaman Crawling Data",
     "Halaman Preprocessing Data", "Halaman Term Frequency","Halaman K Means","Halaman lda knn kmeans", "Halaman Topic Modelling"])

# Menampilkan konten halaman yang dipilih
if selected_page == "Halaman Utama":
    main_page()
elif selected_page == "Halaman Crawling Data":
    crawling_data_page()
elif selected_page == "Halaman Preprocessing Data":
    preprocessing_data_page()
elif selected_page == "Halaman Term Frequency":
    term_frequency_page()
elif selected_page == "Halaman K Means":
    K_Means_page()
# elif selected_page == "Halaman lda knn":
#     lda_knn_page()
elif selected_page == "Halaman lda knn kmeans":
    lda_knn_kmeans_page()
else:
    topic_modelling_page()
