import streamlit as st
from main import main as main_page
from crawling_berita import crawling_berita_page
from preprocessing import preprocessing_page
from cbow import cbow_page
from bilstm import main_app_interface as bilstm_page

selected_page = st.sidebar.selectbox(
    "Pilih Halaman:",
    ["Halaman Utama", "Halaman crawling_berita", "Halaman preprocessing", "Halaman CBOW", "Halaman BILSTM"]
)

if selected_page == "Halaman Utama":
    main_page()
elif selected_page == "Halaman crawling_berita":
    crawling_berita_page()
elif selected_page == "Halaman preprocessing":
    preprocessing_page()
elif selected_page == "Halaman CBOW":
    cbow_page()
elif selected_page == "Halaman BILSTM":
    bilstm_page()
