import streamlit as st
import pandas as pd
import os

def cbow_page():
    st.title("📄 Tampilkan Embedding dari CSV")

    filename = "review_CBOW_weights1 (1).csv"

    if not os.path.exists(filename):
        st.error(f"❌ File '{filename}' tidak ditemukan.")
        st.info("Pastikan file .csv berada di direktori yang sama dengan script ini.")
        return

    try:
        df = pd.read_csv(filename, index_col=0)

        st.success(f"✅ Berhasil memuat '{filename}'")
        st.write(f"Ukuran data: {df.shape[0]} kata × {df.shape[1]} dimensi")

        st.subheader("📊 Tampilkan Embedding Pertama")
        st.dataframe(df.iloc[:5, :10].style.format("{:.6f}"))

        st.download_button(
            label="⬇️ Download CSV",
            data=df.to_csv().encode("utf-8"),
            file_name="review_CBOW_weights_output.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Gagal memuat file CSV: {e}")

if __name__ == "__main__":
    st.set_page_config(page_title="CBOW CSV Viewer", layout="wide", page_icon="📄")
    cbow_csv_viewer()
