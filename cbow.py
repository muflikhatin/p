import streamlit as st
import pandas as pd
import os

def cbow_page():
    st.title("📄 Tampilkan Embedding dari CSV")

    filename = "review_CBOW.pkl"
    
    # Jika file .pkl tidak ditemukan, coba cari file .csv
    if not os.path.exists(filename):
        st.warning(f"File '{filename}' tidak ditemukan.")
        # Coba cari file CSV sebagai alternatif
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'cbow' in f.lower()]
        if csv_files:
            filename = csv_files[0]
            st.info(f"Menggunakan file '{filename}' sebagai alternatif.")
        else:
            st.error("Tidak ditemukan file embedding yang sesuai.")
            st.info("Pastikan file CSV atau PKL berada di direktori yang sama dengan script ini.")
            return

    try:
        # Baca file berdasarkan ekstensi
        if filename.endswith('.pkl'):
            df = pd.read_pickle(filename)
        else:  # CSV
            df = pd.read_csv(filename, index_col=0)

        st.success(f"✅ Berhasil memuat '{filename}'")
        st.write(f"Ukuran data: {df.shape[0]} kata × {df.shape[1]} dimensi")

        # Tampilkan statistik dasar
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jumlah Kata", df.shape[0])
        with col2:
            st.metric("Dimensi Embedding", df.shape[1])
        with col3:
            st.metric("Tipe Data", str(df.dtypes[0]))

        # Fitur pencarian kata
        st.subheader("🔍 Cari Kata")
        search_word = st.text_input("Masukkan kata untuk dilihat vektornya:").strip()
        
        if search_word:
            if search_word in df.index:
                st.write(f"Vektor untuk kata '{search_word}':")
                st.dataframe(df.loc[search_word:search_word].style.format("{:.6f}"))
            else:
                st.warning(f"Kata '{search_word}' tidak ditemukan dalam vocabulary.")

        st.subheader("📊 Tampilan Embedding")
        
        # Pilihan jumlah baris dan kolom yang ditampilkan
        rows = st.slider("Jumlah baris", 5, 50, 10)
        cols = st.slider("Jumlah kolom", 5, min(20, df.shape[1]), 10)
        
        # Tampilkan data
        st.dataframe(df.iloc[:rows, :cols].style.format("{:.6f}"), height=400)

        # Tombol unduh
        st.download_button(
            label="⬇️ Download CSV",
            data=df.to_csv().encode("utf-8"),
            file_name="review_CBOW_weights_output.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Gagal memuat file: {e}")
        st.info("Pastikan format file benar (CSV dengan index atau PKL).")

if __name__ == "__main__":
    st.set_page_config(page_title="CBOW CSV Viewer", layout="wide", page_icon="📄")
    cbow_page()  # Memperbaiki pemanggilan fungsi yang salah
