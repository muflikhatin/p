import streamlit as st
import pandas as pd
import os
import gdown
import pickle
import numpy as np
import joblib

@st.cache_resource
def load_cbow_model():
    drive_id = "1no16BOGbgbgMWENU_ZHmka1rEIXUMJfJ"
    filename = "review_CBOW.pkl"
    
    if not os.path.exists(filename):
        with st.spinner('Mengunduh model CBOW dari Google Drive...'):
            url = f"https://drive.google.com/uc?id={drive_id}"
            try:
                gdown.download(url, filename, quiet=False)
                if os.path.exists(filename):
                    st.success("File berhasil diunduh!")
                else:
                    st.error("File gagal diunduh. Pastikan ID benar dan file dapat diakses publik.")
                    return None
            except Exception as e:
                st.error(f"Gagal mengunduh file: {e}")
                return None
    
    try:
        # Coba load dengan joblib pertama (lebih kompatibel)
        try:
            model = joblib.load(filename)
            st.success("Model berhasil dimuat dengan joblib!")
            return model
        except Exception as e:
            st.warning(f"Joblib gagal: {e}. Mencoba dengan pickle...")
            
            # Baca file sebagai binary
            with open(filename, 'rb') as f:
                data = f.read()
            
            # Perbaiki masalah kompatibilitas numpy._core jika ada
            if b'numpy._core' in data:
                st.warning("Memperbaiki kompatibilitas NumPy...")
                data = data.replace(b'numpy._core', b'numpy.core')
            
            # Load data yang sudah diperbaiki
            model = pickle.loads(data)
            st.success("Model berhasil dimuat dengan pickle!")
            return model
            
    except Exception as e:
        st.error(f"Error memuat model: {str(e)}")
        
        # Coba metode alternatif untuk file yang korup atau format berbeda
        try:
            st.info("Mencoba metode alternatif...")
            
            # Coba load sebagai file numpy
            try:
                model = np.load(filename, allow_pickle=True)
                st.success("Model berhasil dimuat sebagai file numpy!")
                return model
            except:
                pass
                
            # Coba baca sebagai bytes dan cari format yang sesuai
            with open(filename, 'rb') as f:
                first_bytes = f.read(100)  # Baca 100 byte pertama
                
            # Cek jika ini adalah file gensim
            if b'gensim' in first_bytes or b'word2vec' in first_bytes.lower():
                st.info("File terdeteksi sebagai model gensim. Mencoba load dengan gensim...")
                try:
                    from gensim.models import Word2Vec, KeyedVectors
                    model = Word2Vec.load(filename)
                    st.success("Model berhasil dimuat dengan gensim!")
                    return model
                except:
                    try:
                        model = KeyedVectors.load(filename)
                        st.success("Model berhasil dimuat sebagai KeyedVectors!")
                        return model
                    except:
                        pass
                        
        except Exception as alt_e:
            st.error(f"Metode alternatif juga gagal: {alt_e}")
            
        return None

def cbow_page():
    st.title("📄 Tampilkan Embedding dari CBOW Model")
    
    # Informasi tentang aplikasi
    st.info("""
    Aplikasi ini memuat model CBOW dari Google Drive dan menampilkan embedding vektornya.
    Pastikan file model memiliki izin akses publik di Google Drive.
    """)
    
    # Memuat model CBOW
    with st.spinner('Memuat model CBOW...'):
        model = load_cbow_model()
    
    if model is None:
        st.error("❌ Gagal memuat model CBOW.")
        st.info("""
        Beberapa kemungkinan penyebab:
        1. File tidak ditemukan di Google Drive
        2. Izin akses file tidak diatur ke 'Anyone with the link can view'
        3. Format file tidak kompatibel
        4. File mungkin korup atau dalam format yang tidak didukung
        """)
        
        # Tampilkan informasi file jika ada
        if os.path.exists("review_CBOW.pkl"):
            file_size = os.path.getsize("review_CBOW.pkl")
            st.write(f"📁 File ditemukan dengan ukuran: {file_size} bytes")
            
            # Baca beberapa byte pertama untuk analisis
            with open("review_CBOW.pkl", 'rb') as f:
                first_bytes = f.read(20)
                st.write(f"🔍 Byte pertama: {first_bytes}")
        
        return
    
    st.success("✅ Berhasil memuat model CBOW")
    
    # Deteksi tipe model dan tampilkan embedding
    try:
        df = None
        
        # Jika model adalah gensim Word2Vec
        if hasattr(model, 'wv'):
            st.info("Model terdeteksi sebagai gensim Word2Vec")
            words = list(model.wv.key_to_index.keys())[:100]  # Ambil 100 kata pertama
            vectors = [model.wv[word] for word in words]
            df = pd.DataFrame(vectors, index=words)
            
        # Jika model adalah KeyedVectors (gensim)
        elif hasattr(model, 'key_to_index'):
            st.info("Model terdeteksi sebagai gensim KeyedVectors")
            words = list(model.key_to_index.keys())[:100]
            vectors = [model[word] for word in words]
            df = pd.DataFrame(vectors, index=words)
            
        # Jika model adalah Keras/TensorFlow
        elif hasattr(model, 'get_weights'):
            st.info("Model terdeteksi sebagai Keras/TensorFlow")
            embeddings = model.get_weights()[0]
            df = pd.DataFrame(embeddings)
            
        # Jika model adalah array numpy
        elif isinstance(model, np.ndarray):
            st.info("Model terdeteksi sebagai array NumPy")
            df = pd.DataFrame(model)
            
        # Jika model adalah dictionary (format embedding sederhana)
        elif isinstance(model, dict):
            st.info("Model terdeteksi sebagai dictionary")
            words = list(model.keys())[:100]
            vectors = list(model.values())[:100]
            df = pd.DataFrame(vectors, index=words)
            
        else:
            st.error(f"Format model tidak dikenali. Tipe: {type(model)}")
            # Coba akses sebagai objek dengan atribut tertentu
            try:
                # Jika model memiliki atribut 'vectors'
                if hasattr(model, 'vectors'):
                    df = pd.DataFrame(model.vectors)
                    st.info("Menggunakan atribut 'vectors' dari model")
                else:
                    # Coba konversi langsung ke DataFrame
                    df = pd.DataFrame(model)
                    st.info("Model dikonversi langsung ke DataFrame")
            except:
                st.error("Tidak dapat mengonversi model ke format yang dapat ditampilkan")
                return
        
        # Tampilkan informasi embedding
        st.write(f"📊 Ukuran embedding: {df.shape[0]} baris × {df.shape[1]} kolom")
        
        # Pilihan untuk menampilkan lebih banyak data
        n_rows = st.slider("Jumlah baris yang ditampilkan", 5, 50, 10)
        n_cols = st.slider("Jumlah kolom yang ditampilkan", 5, min(20, df.shape[1]), 10)
        
        st.subheader(f"🔍 Preview Embedding ({n_rows} baris pertama, {n_cols} kolom pertama)")
        st.dataframe(df.iloc[:n_rows, :n_cols].style.format("{:.6f}"))
        
        # Informasi statistik
        st.subheader("📈 Statistik Embedding")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rata-rata", f"{df.values.mean():.6f}")
            st.metric("Standar Deviasi", f"{df.values.std():.6f}")
        with col2:
            st.metric("Nilai Minimum", f"{df.values.min():.6f}")
            st.metric("Nilai Maksimum", f"{df.values.max():.6f}")
        
        # Opsi download
        st.subheader("💾 Download Embedding")
        csv = df.to_csv()
        st.download_button(
            label="⬇️ Download sebagai CSV",
            data=csv,
            file_name="review_CBOW_embeddings.csv",
            mime="text/csv",
            help="Download seluruh data embedding dalam format CSV"
        )
        
    except Exception as e:
        st.error(f"Gagal memproses model: {str(e)}")
        st.info("Model mungkin memiliki format yang tidak terduga.")

if __name__ == "__main__":
    st.set_page_config(
        page_title="CBOW Embedding Viewer", 
        layout="wide", 
        page_icon="📄",
        initial_sidebar_state="expanded"
    )
    
    # Tambahkan sidebar untuk informasi tambahan
    with st.sidebar:
        st.title("ℹ️ Informasi Aplikasi")
        st.write("""
        Aplikasi ini menampilkan embedding vektor dari model CBOW yang disimpan di Google Drive.
        
        **Cara penggunaan:**
        1. Pastikan file model ada di Google Drive
        2. Atur izin file ke "Anyone with the link can view"
        3. Ganti drive_id dengan ID file Anda
        4. Jalankan aplikasi
        """)
        
        st.divider()
        st.write("**Versi Library:**")
        st.write(f"- Streamlit: {st.__version__}")
        st.write(f"- Pandas: {pd.__version__}")
        st.write(f"- NumPy: {np.__version__}")
    
    cbow_page()
