import streamlit as st
import pandas as pd
import os
import gdown
import pickle

@st.cache_resource
def load_cbow_model():
    drive_id = "1no16BOGbgbgMWENU_ZHmka1rEIXUMJfJ"  # Ganti dengan ID file review_CBOW.pkl Anda
    filename = "review_CBOW.pkl"
    
    if not os.path.exists(filename):
        with st.spinner('Mengunduh model CBOW dari Google Drive...'):
            url = f"https://drive.google.com/uc?id={drive_id}"
            gdown.download(url, filename, quiet=False)
    
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None

def cbow_page():
    st.title("📄 Tampilkan Embedding dari CBOW Model")

    # Memuat model CBOW
    model = load_cbow_model()
    
    if model is None:
        st.error("❌ Gagal memuat model CBOW.")
        st.info("Pastikan file 'review_CBOW.pkl' tersedia di Google Drive dengan ID yang benar.")
        return
    
    st.success("✅ Berhasil memuat model CBOW")
    
    # Cek jika model memiliki weights/embeddings
    if hasattr(model, 'wv'):
        # Jika menggunakan gensim Word2Vec
        words = list(model.wv.key_to_index.keys())[:100]  # Ambil 100 kata pertama
        vectors = [model.wv[word] for word in words]
        df = pd.DataFrame(vectors, index=words)
        
        st.write(f"Ukuran embedding: {df.shape[0]} kata × {df.shape[1]} dimensi")
        
        st.subheader("📊 Tampilkan 5×10 Embedding Pertama")
        st.dataframe(df.iloc[:5, :10].style.format("{:.6f}"))
        
        # Konversi ke CSV untuk download
        csv = df.to_csv()
        
        st.download_button(
            label="⬇️ Download Embedding sebagai CSV",
            data=csv,
            file_name="review_CBOW_embeddings.csv",
            mime="text/csv"
        )
        
    elif hasattr(model, 'get_weights'):
        # Jika menggunakan Keras/TensorFlow model
        embeddings = model.get_weights()[0]
        df = pd.DataFrame(embeddings)
        
        st.write(f"Ukuran embedding: {df.shape[0]} kata × {df.shape[1]} dimensi")
        
        st.subheader("📊 Tampilkan 5×10 Embedding Pertama")
        st.dataframe(df.iloc[:5, :10].style.format("{:.6f}"))
        
        # Konversi ke CSV untuk download
        csv = df.to_csv()
        
        st.download_button(
            label="⬇️ Download Embedding sebagai CSV",
            data=csv,
            file_name="review_CBOW_embeddings.csv",
            mime="text/csv"
        )
    else:
        st.error("Format model tidak dikenali. Pastikan model adalah Word2Vec gensim atau model Keras dengan embedding layer.")

if __name__ == "__main__":
    st.set_page_config(page_title="CBOW Embedding Viewer", layout="wide", page_icon="📄")
    cbow_page()
