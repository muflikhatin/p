import os
import re
import pickle
import warnings
from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ========================#
#    PAGE CONFIG          #
# ========================#
st.set_page_config(page_title="Doc Classifier", layout="wide")
warnings.filterwarnings("ignore")

# ========================#
#   CONSTANTS             #
# ========================#
CLASS_NAMES = ['Travel', 'Edukasi', 'Sports', 'Politik', 'Health']
MODEL_PATH = "best_model_15epochs.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_SEQUENCE_LENGTH = 300
CHUNK_SIZE = 500   # jumlah dokumen diproses per chunk

# ========================#
#   LOAD MODEL/TOKENIZER  #
# ========================#
@st.cache_resource
def load_tokenizer(path: str = TOKENIZER_PATH):
    """Load tokenizer sekali saja (cache_resource untuk object besar)."""
    try:
        with open(path, "rb") as f:
            tokenizer = pickle.load(f)
        return tokenizer
    except Exception as e:
        st.error(f"Tokenizer gagal dimuat: {e}")
        return None

@st.cache_resource
def load_model(path: str = MODEL_PATH):
    """Load model sekali saja (cache_resource)."""
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        st.error(f"Model gagal dimuat: {e}")
        return None

# ========================#
#   TEXT PREPROCESSING    #
# ========================#
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = re.sub(r"[^\w\s]", "", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()

def preprocess_text(text: str, tokenizer):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding="post")

def map_category(kat: str) -> str:
    mapping = {
        'travel': 'Travel', 'edukasi': 'Edukasi', 'pendidikan': 'Edukasi',
        'sports': 'Sports', 'olahraga': 'Sports', 'politik': 'Politik',
        'health': 'Health', 'kesehatan': 'Health'
    }
    return mapping.get(str(kat).strip().lower(), 'Unknown')

# ========================#
#   FILE READING UTILS    #
# ========================#
def process_chunk(chunk: pd.DataFrame, tokenizer):
    try:
        chunk.columns = [c.strip() for c in chunk.columns]
        if 'Kategori' not in chunk.columns or 'Konten' not in chunk.columns:
            return None

        chunk['Kategori'] = chunk['Kategori'].map(map_category)
        chunk = chunk[chunk['Kategori'].isin(CLASS_NAMES)]
        chunk['Konten'] = chunk['Konten'].astype(str).apply(clean_text)
        chunk['Padded'] = chunk['Konten'].apply(lambda x: preprocess_text(x, tokenizer))
        return chunk.dropna(subset=['Padded'])
    except Exception:
        return None

def read_data_in_chunks(uploaded_file, tokenizer):
    try:
        filename = uploaded_file.name.lower()
        full_df = pd.DataFrame()
        progress = st.progress(0)
        status = st.empty()

        if filename.endswith(".csv"):
            # Hitung total chunk untuk progress bar
            total_chunks = sum(1 for _ in pd.read_csv(uploaded_file, encoding="utf-8",
                                                     on_bad_lines="skip", chunksize=CHUNK_SIZE))
            uploaded_file.seek(0)  # reset pointer
            for i, chunk in enumerate(pd.read_csv(uploaded_file, encoding="utf-8",
                                                  on_bad_lines="skip", chunksize=CHUNK_SIZE)):
                status.text(f"Processing chunk {i+1}/{total_chunks}")
                processed = process_chunk(chunk, tokenizer)
                if processed is not None:
                    full_df = pd.concat([full_df, processed], ignore_index=True)
                progress.progress((i+1)/total_chunks)

        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
            total_rows = len(df)
            num_chunks = (total_rows // CHUNK_SIZE) + 1
            for i in range(num_chunks):
                start, end = i * CHUNK_SIZE, min((i + 1) * CHUNK_SIZE, total_rows)
                status.text(f"Processing rows {start+1}-{end} of {total_rows}")
                processed = process_chunk(df.iloc[start:end], tokenizer)
                if processed is not None:
                    full_df = pd.concat([full_df, processed], ignore_index=True)
                progress.progress((i+1)/num_chunks)

        elif filename.endswith(".txt"):
            df = pd.read_csv(uploaded_file, delimiter="\t", encoding="utf-8", on_bad_lines="skip")
            total_rows = len(df)
            num_chunks = (total_rows // CHUNK_SIZE) + 1
            for i in range(num_chunks):
                start, end = i * CHUNK_SIZE, min((i + 1) * CHUNK_SIZE, total_rows)
                status.text(f"Processing rows {start+1}-{end} of {total_rows}")
                processed = process_chunk(df.iloc[start:end], tokenizer)
                if processed is not None:
                    full_df = pd.concat([full_df, processed], ignore_index=True)
                progress.progress((i+1)/num_chunks)

        else:
            st.error("Unsupported file type. Upload CSV, Excel, or TXT.")
            return None

        progress.empty()
        status.text("Processing completed!")
        return full_df

    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# ========================#
#   PREDICTION UTILS      #
# ========================#
def predict_in_batches(df, model):
    try:
        padded_values = df["Padded"].values
        batch_size = 100
        num_batches = (len(padded_values) // batch_size) + 1

        preds_all, conf_all = [], []
        progress = st.progress(0)
        status = st.empty()

        for i in range(num_batches):
            start, end = i * batch_size, min((i + 1) * batch_size, len(padded_values))
            status.text(f"Predicting batch {i+1}/{num_batches}")
            batch = np.vstack(padded_values[start:end])
            preds = model.predict(batch, verbose=0)
            preds_all.extend([CLASS_NAMES[np.argmax(p)] for p in preds])
            conf_all.extend([np.max(p) for p in preds])
            progress.progress((i + 1) / num_batches)

        df["Prediksi"] = preds_all
        df["Confidence"] = conf_all
        df["Correct"] = df["Kategori"] == df["Prediksi"]

        progress.empty()
        status.text("Prediction completed!")
        return df[["Kategori", "Konten", "Prediksi", "Confidence", "Correct"]]

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return pd.DataFrame()

# ========================#
#   DISPLAY RESULTS       #
# ========================#
def display_results(result_df, sample_size=10):
    accuracy = result_df["Correct"].mean()
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{accuracy:.1%}")
    c2.metric("Total Documents", len(result_df))

    st.subheader("Sample Results")
    sample_df = result_df.sample(n=min(sample_size, len(result_df)), random_state=42)
    st.dataframe(sample_df.style.format({'Confidence': '{:.1%}'}))

    # Confusion Matrix
    if not result_df.empty:
        st.subheader("Confusion Matrix")
        cm = pd.crosstab(result_df["Kategori"], result_df["Prediksi"],
                         rownames=["Actual"], colnames=["Predicted"], margins=True)
        st.dataframe(cm.style.background_gradient(cmap="Blues"))

def predict_single_text(text, model, tokenizer):
    padded = preprocess_text(text, tokenizer)
    pred = model.predict(padded, verbose=0)[0]
    return CLASS_NAMES[np.argmax(pred)], np.max(pred), pred

# ========================#
#       MAIN PAGE         #
# ========================#
def bilstm_page():
    st.title("📊 Document Classification (Large Files Support)")
    with st.spinner("Loading model and tokenizer..."):
        model = load_model()
        tokenizer = load_tokenizer()
        if model is None or tokenizer is None:
            st.stop()

    mode = st.radio("Choose Mode", ["📄 Upload File", "🔍 Single Prediction"], horizontal=True)

    # ----------- Upload File Mode ----------- #
    if mode == "📄 Upload File":
        st.info("Supports large files with progress tracking.")
        uploaded = st.file_uploader("Upload CSV, Excel, or TXT", type=['csv', 'xlsx', 'xls', 'txt'])
        if uploaded:
            with st.spinner("Reading and processing file..."):
                df = read_data_in_chunks(uploaded, tokenizer)
            if df is not None and not df.empty:
                st.success(f"Processed {len(df)} documents. Predicting...")
                result_df = predict_in_batches(df, model)
                if not result_df.empty:
                    display_results(result_df)

                    # ---- Download Buttons ----
                    st.subheader("Download Results")
                    excel_buf = BytesIO()
                    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                        result_df.to_excel(writer, index=False, sheet_name="Results")
                    st.download_button("Download Excel",
                        excel_buf.getvalue(),
                        "document_classification_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    st.download_button("Download CSV",
                        result_df.to_csv(index=False).encode("utf-8"),
                        "document_classification_results.csv",
                        mime="text/csv")

    # ----------- Single Prediction Mode ----------- #
    else:
        text = st.text_area("Enter text to classify:", height=200)
        if st.button("Classify") and text.strip():
            pred_class, conf, dist = predict_single_text(text, model, tokenizer)
            st.success(f"Predicted class: **{pred_class}** (confidence: {conf:.1%})")
            st.bar_chart(pd.DataFrame({"Class": CLASS_NAMES, "Confidence": dist}).set_index("Class"))

# ========================#
#   ENTRY POINT           #
# ========================#
if __name__ == "__main__":
    bilstm_page()
