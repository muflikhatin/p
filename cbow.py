import streamlit as st
import pandas as pd
import os
import gdown
import pickle
import numpy as np

@st.cache_resource
def load_cbow_model():
    drive_id = "1no16BOGbgbgMWENU_ZHmka1rEIXUMJfJ"
    filename = "review_CBOW.pkl"
    
    if not os.path.exists(filename):
        with st.spinner('Mengunduh model CBOW dari Google Drive...'):
            url = f"https://drive.google.com/uc?id={drive_id}"
            gdown.download(url, filename, quiet=False)
    
    try:
        # Metode khusus untuk menangani error numpy._core
        with open(filename, 'rb') as f:
            # Baca file sebagai bytes
            model_data = f.read()
            
            # Coba perbaiki masalah numpy._core
            if b'numpy._core' in model_data:
                st.warning("Memperbaiki kompatibilitas NumPy...")
                # Ganti numpy._core dengan numpy.core
                model_data = model_data.replace(b'numpy._core', b'numpy.core')
            
            # Load dari bytes yang sudah diperbaiki
            model = pickle.loads(model_data)
            
        return model
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        # Coba metode alternatif
        try:
            import joblib
            model = joblib.load(filename)
            return model
        except:
            return None

def cbow_page():
    st.title("📄 Tampilkan Embedding dari CBOW Model")

    # Memuat model CBOW
    model = load_cbow_model()
    
    if model is None:
        st.error("❌ Gagal memuat model CBOW.")
        st.info("""
        Pastikan:
        1. File 'review_CBOW.pkl' tersedia di Google Drive dengan ID yang benar
        2. File memiliki izin akses publik (Anyone with the link can view)
        3. Model kompatibel dengan versi library yang digunakan
        """)
        return
    
    st.success("✅ Berhasil memuat model CBOW")
    
    # Deteksi tipe model dan tampilkan embedding
    try:
        # Jika model adalah gensim Word2Vec
        if hasattr(model, 'wv'):
            words = list(model.wv.key_to_index.keys())[:100]
            vectors = [model.wv[word] for word in words]
            df = pd.DataFrame(vectors, index=words)
            
        # Jika model adalah Keras/TensorFlow
        elif hasattr(model, 'get_weights'):
            embeddings = model.get_weights()[0]
            df = pd.DataFrame(embeddings)
            
        # Jika model adalah array numpy
        elif isinstance(model, np.ndarray):
            df = pd.DataFrame(model)
            
        # Jika model adalah dictionary (format embedding sederhana)
        elif isinstance(model, dict):
            words = list(model.keys())[:100]
            vectors = list(model.values())[:100]
            df = pd.DataFrame(vectors, index=words)
            
        else:
            st.error(f"Format model tidak dikenali. Tipe: {type(model)}")
            return
        
        # Tampilkan informasi embedding
        st.write(f"📊 Ukuran embedding: {df.shape[0]} kata × {df.shape[1]} dimensi")
        
        st.subheader("🔍 Preview Embedding (5 kata pertama, 10 dimensi pertama)")
        st.dataframe(df.iloc[:5, :10].style.format("{:.6f}"))
        
        # Opsi download
        csv = df.to_csv()
        st.download_button(
            label="⬇️ Download Embedding sebagai CSV",
            data=csv,
            file_name="review_CBOW_embeddings.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Gagal memproses model: {e}")
        st.info("Model mungkin memiliki format yang tidak terduga.")

if __name__ == "__main__":
    st.set_page_config(page_title="CBOW Embedding Viewer", layout="wide", page_icon="📄")
    cbow_page()
