# app.py
import streamlit as st
from main import main as main_page
from crowling_data import main as crawling_data_page
from preprocessing_data import main as preprocessing_data_page
from term_frequency import main as term_frequency_page
from K_Means import main as K_Means_page
from topic_modelling import main as topic_modelling_page
from lda_knn_kmeans import main as lda_knn_kmeans_page
from closeness_centrality_crawling import main as closeness_centrality_crawling_page
from closeness_centrality import main as closeness_centrality_page
from klasivikasi_SVM import main as klasivikasi_SVM_page

# Sidebar untuk navigasi
selected_page = st.sidebar.selectbox(
    "Pilih Halaman:",
    ["Halaman Utama", "Halaman Crawling Data",
     "Halaman Preprocessing Data", "Halaman Term Frequency","Halaman K Means","Halaman lda knn kmeans", "Halaman Topic Modelling", "Halaman closeness centrality crawling", "Halaman closeness centrality", "Halaman Klasifikasi_SVM"])

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
elif selected_page == "Halaman lda knn":
    lda_knn_page()
elif selected_page == "Halaman closeness centrality crawling":
    closeness_centrality_crawling_page()
elif selected_page == "Halaman closeness centrality":
    closeness_centrality_page()
elif selected_page == "Halaman Klasifikasi_SVM":
    klasivikasi_SVM_page()    
else:
    topic_modelling_page()
