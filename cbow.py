import streamlit as st
import pandas as pd
import os
import gdown

def cbow_page():
    st.title("📄 Tampilkan Embedding dari Pickle (CBOW)")

    # ID file dari Google Drive
    file_id = "1no16BOGbgbgMWENU_ZHmka1rEIXUMJfJ"
    filename = "review_CBOW.pkl"
    url = f"https://drive.google.com/uc?id={file_id}"

    # Jika file belum ada, download dulu
    if not os.path.exists(filename):
        st.info("⬇️ Mengunduh file dari Google Drive...")
        gdown.download(url, filename, quiet=False)

    try:
        # Load dari pickle (bukan CSV)
        df = pd.read_pickle(filename)

        st.success(f"✅ Berhasil memuat '{filename}'")
        st.write(f"Ukuran data: {df.shape[0]} kata × {df.shape[1]} dimensi")

        st.subheader("📊 Tampilkan 5×10 Embedding Pertama")
        st.dataframe(df.iloc[:5, :10].style.format("{:.6f}"))

        # Simpan ke CSV untuk bisa diunduh user
        st.download_button(
            label="⬇️ Download CSV",
            data=df.to_csv().encode("utf-8"),
            file_name="review_CBOW_weights_output.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Gagal memuat file: {e}")

if __name__ == "__main__":
    st.set_page_config(page_title="CBOW Embedding Viewer", layout="wide", page_icon="📄")
    cbow_page()
