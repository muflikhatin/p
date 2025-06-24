import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocessing_page():
    st.title("📊 Preprocessing Data Teks")
    
    if os.path.exists("df.csv"):
        try:
            # 1. Load and prepare data
            df = pd.read_csv("df.csv", encoding='utf-8')
            
            # Category mapping
            kategori_mapping = {'Travel': 0, 'Edukasi': 1, 'Sports': 2, 'Politik': 3, 'Health': 4}
            reverse_mapping = {v: k for k, v in kategori_mapping.items()}

            if 'Label' not in df.columns and 'Kategori' in df.columns:
                df['Label'] = df['Kategori'].replace(kategori_mapping)

            st.subheader("1. Data Awal")
            st.dataframe(df.head())

            # 2. Show category distribution
            if 'Label' in df.columns:
                st.subheader("2. Distribusi Kategori")
                label_counts = df['Label'].value_counts().sort_index()

                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.bar(label_counts.index, label_counts.values, color='skyblue')
                ax.set_title("Distribusi Kategori")
                ax.set_xlabel("Kategori")
                ax.set_ylabel("Jumlah")
                ax.grid(axis='y')
                ax.set_xticks(list(label_counts.index))
                ax.set_xticklabels([reverse_mapping[i] for i in label_counts.index], rotation=45)

                for bar in bars:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            int(bar.get_height()), ha='center', va='bottom')

                st.pyplot(fig)

                st.markdown("**Keterangan Label Kategori:**")
                for kategori, kode in kategori_mapping.items():
                    st.markdown(f"- **{kode}** = {kategori}")

            # 3. Preprocessing Section
            st.subheader("3. Preprocessing Teks")
            
            # Column selection
            cols_to_drop = st.multiselect(
                "Pilih kolom yang akan dihapus:",
                df.columns,
                default=['Judul', 'Tanggal', 'Jumlah_Kata', 'Link']
            )
            
            # Hardcoded preprocessing parameters
            max_length = 300  # Fixed sequence length
            min_word_count = 3  # Minimum word count threshold
            remove_short = True  # Always remove short texts
            
            if st.button("Lakukan Preprocessing"):
                with st.spinner('Sedang memproses...'):
                    # 1. Drop unused columns
                    df_clean = df.drop(cols_to_drop, axis=1)
                    
                    # 2. Text cleaning
                    def clean_text(text):
                        if pd.isna(text):
                            return ""
                        text = re.sub(r'[^\w\s]', ' ', str(text))  # Remove punctuation
                        text = re.sub(r'\d+', '', text)  # Remove numbers
                        text = text.lower().strip()
                        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                        return text
                    
                    df_clean['Konten (clean)'] = df_clean['Konten'].apply(clean_text)
                    
                    # 3. Filter short texts
                    if remove_short:
                        initial_count = len(df_clean)
                        df_clean['word_count'] = df_clean['Konten (clean)'].apply(lambda x: len(x.split()))
                        df_clean = df_clean[df_clean['word_count'] >= min_word_count]
                        removed_count = initial_count - len(df_clean)
                        st.info(f"Menghapus {removed_count} dokumen dengan word count < {min_word_count}")
                    
                    # 4. Tokenization with special padding token
                    tokenizer = Tokenizer(oov_token="<OOV>", filters='')
                    tokenizer.fit_on_texts(df_clean['Konten (clean)'])
                    
                    # Add explicit padding token at index 0
                    tokenizer.word_index = {k:(v+1) for k,v in tokenizer.word_index.items()}
                    tokenizer.word_index["<PAD>"] = 0
                    tokenizer.index_word = {v:k for k,v in tokenizer.word_index.items()}
                    
                    vocab_size = len(tokenizer.word_index)
                    
                    # 5. Convert to sequences
                    sequences = tokenizer.texts_to_sequences(df_clean['Konten (clean)'])
                    
                    # 6. Padding with explicit <PAD> token (0)
                    padded_sequences = pad_sequences(
                        sequences,
                        maxlen=max_length,
                        padding='post',
                        truncating='post',
                        value=tokenizer.word_index["<PAD>"]
                    )
                    
                    # Verify all sequences have length 300
                    assert all(len(seq) == max_length for seq in padded_sequences), "Padding failed!"
                    
                    # 7. Create text representations
                    df_clean['Tokenized'] = [
                        [tokenizer.index_word.get(idx, '<OOV>') for idx in seq if idx != 0] 
                        for seq in sequences
                    ]
                    
                    df_clean['Padded_Sequence'] = [
                        [tokenizer.index_word.get(idx, '<OOV>' if idx != 0 else '<PAD>') 
                         for idx in seq] 
                        for seq in padded_sequences
                    ]
                    
                    df_clean['Padded_Numerical'] = list(padded_sequences)
                    
                    # 8. Display results
                    st.subheader("Hasil Preprocessing")
                    
                    with st.expander("Lihat Data Preprocessing", expanded=True):
                        st.dataframe(df_clean[['Konten', 'Konten (clean)', 'Tokenized', 'Padded_Sequence']].head())
                    
                    with st.expander("Verifikasi Padding"):
                        st.write("Contoh pertama:")
                        st.json({
                            "Original": df_clean['Konten'].iloc[0],
                            "Cleaned": df_clean['Konten (clean)'].iloc[0],
                            "Tokenized": df_clean['Tokenized'].iloc[0],
                            "Padded_Sequence": df_clean['Padded_Sequence'].iloc[0],
                            "Padded_Numerical": df_clean['Padded_Numerical'].iloc[0].tolist(),
                            "Length": len(df_clean['Padded_Numerical'].iloc[0])
                        })
                        
                        st.success(f"Semua sequence memiliki panjang {max_length} setelah padding")
                        st.write(f"Shape padded sequences: {padded_sequences.shape}")
                    
                    # 9. Save results
                    st.subheader("Simpan Hasil")
                    
                    # Save CSV
                    csv = df_clean.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download CSV",
                        csv,
                        "preprocessed_data.csv",
                        "text/csv"
                    )
                    
                    # Save tokenizer
                    with open('tokenizer.pkl', 'wb') as f:
                        pickle.dump(tokenizer, f)
                    
                    # Save numpy arrays
                    np.save('padded_sequences.npy', padded_sequences)
                    if 'Label' in df_clean.columns:
                        np.save('labels.npy', df_clean['Label'].values)
                    
                    st.success("Preprocessing selesai! Data berhasil disimpan.")
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("File 'df.csv' tidak ditemukan.")

if __name__ == "__main__":
    preprocessing_page()
