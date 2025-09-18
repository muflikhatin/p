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
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Set page config before anything else
st.set_page_config(page_title="Doc Classifier", layout="wide")

warnings.filterwarnings('ignore')

# Constants
CLASS_NAMES = ['Travel', 'Edukasi', 'Sports', 'Politik', 'Health']
MODEL_PATH = "best_model_15epochs.h5"
TOKENIZER_PATH = "tokenizer.json"  # Changed from .pkl to .json
MAX_SEQUENCE_LENGTH = 300
CHUNK_SIZE = 500  # Number of documents to process at a time

# Global variables to store error messages
model_error = None
tokenizer_error = None

@st.cache_resource
def load_tokenizer(path=TOKENIZER_PATH):
    global tokenizer_error
    tokenizer_error = None
    try:
        if not os.path.exists(path):
            tokenizer_error = f"File tokenizer tidak ditemukan: {path}"
            return None
            
        with open(path, 'r') as f:
            tokenizer_data = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_data)
        return tokenizer
    except Exception as e:
        tokenizer_error = f"Error loading tokenizer: {str(e)}"
        return None

@st.cache_resource
def load_model(model_path=MODEL_PATH):
    global model_error
    model_error = None
    try:
        if not os.path.exists(model_path):
            model_error = f"File model tidak ditemukan: {model_path}"
            return None
            
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        model_error = f"Error loading model: {str(e)}"
        return None

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text, tokenizer):
    try:
        text = clean_text(text)
        sequences = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        return padded
    except Exception as e:
        st.warning(f"Preprocessing failed: {str(e)}")
        return None

def map_category(kat):
    mapping = {
        'travel': 'Travel', 'edukasi': 'Edukasi', 'pendidikan': 'Edukasi',
        'sports': 'Sports', 'olahraga': 'Sports', 'politik': 'Politik',
        'health': 'Health', 'kesehatan': 'Health'
    }
    kat = str(kat).strip().lower()
    return mapping.get(kat, 'Unknown')

def read_data_in_chunks(uploaded_file, tokenizer):
    try:
        filename = uploaded_file.name.lower()
        
        # Initialize an empty DataFrame to store results
        full_df = pd.DataFrame()
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if filename.endswith('.csv'):
            # Read CSV in chunks
            chunks = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip', chunksize=CHUNK_SIZE)
            total_chunks = sum(1 for _ in pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip', chunksize=CHUNK_SIZE))
            
            for i, chunk in enumerate(chunks):
                status_text.text(f"Processing chunk {i+1} of {total_chunks}...")
                processed_chunk = process_chunk(chunk, tokenizer)
                if processed_chunk is not None and not processed_chunk.empty:
                    full_df = pd.concat([full_df, processed_chunk], ignore_index=True)
                progress_bar.progress((i + 1) / total_chunks)
                
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            # For Excel files, read all at once but process in chunks
            df = pd.read_excel(uploaded_file)
            total_rows = len(df)
            num_chunks = (total_rows // CHUNK_SIZE) + 1
            
            for i in range(num_chunks):
                start_idx = i * CHUNK_SIZE
                end_idx = min((i + 1) * CHUNK_SIZE, total_rows)
                status_text.text(f"Processing rows {start_idx+1} to {end_idx} of {total_rows}...")
                chunk = df.iloc[start_idx:end_idx]
                processed_chunk = process_chunk(chunk, tokenizer)
                if processed_chunk is not None and not processed_chunk.empty:
                    full_df = pd.concat([full_df, processed_chunk], ignore_index=True)
                progress_bar.progress((i + 1) / num_chunks)
                
        elif filename.endswith('.txt'):
            # For TXT files, read all at once but process in chunks
            df = pd.read_csv(uploaded_file, delimiter='\t', encoding='utf-8', on_bad_lines='skip')
            total_rows = len(df)
            num_chunks = (total_rows // CHUNK_SIZE) + 1
            
            for i in range(num_chunks):
                start_idx = i * CHUNK_SIZE
                end_idx = min((i + 1) * CHUNK_SIZE, total_rows)
                status_text.text(f"Processing rows {start_idx+1} to {end_idx} of {total_rows}...")
                chunk = df.iloc[start_idx:end_idx]
                processed_chunk = process_chunk(chunk, tokenizer)
                if processed_chunk is not None and not processed_chunk.empty:
                    full_df = pd.concat([full_df, processed_chunk], ignore_index=True)
                progress_bar.progress((i + 1) / num_chunks)
                
        else:
            st.error("Unsupported file type. Please upload a CSV, Excel (.xlsx), or TXT file.")
            return None

        status_text.text("Processing completed!")
        progress_bar.empty()
        
        return full_df
    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def process_chunk(chunk, tokenizer):
    try:
        chunk.columns = [c.strip() for c in chunk.columns]

        if 'Kategori' not in chunk.columns or 'Konten' not in chunk.columns:
            return None

        chunk['Kategori'] = chunk['Kategori'].map(map_category)
        chunk = chunk[chunk['Kategori'].isin(CLASS_NAMES)]
        chunk['Konten'] = chunk['Konten'].astype(str).apply(clean_text)

        chunk['Padded'] = chunk['Konten'].apply(lambda x: preprocess_text(x, tokenizer))
        chunk = chunk[chunk['Padded'].notna()]
        chunk = chunk[chunk['Padded'].apply(lambda x: x is not None and x.size > 0)]

        return chunk
    
    except Exception as e:
        st.warning(f"Error processing chunk: {str(e)}")
        return None

def predict_in_batches(df, model):
    try:
        # Initialize empty lists to store results
        all_preds = []
        all_confidences = []
        
        # Get the padded values
        padded_values = df['Padded'].values
        
        # Process in batches to avoid memory issues
        batch_size = 100  # Number of documents to predict at once
        num_batches = (len(padded_values) // batch_size) + 1
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(padded_values))
            
            status_text.text(f"Predicting batch {i+1} of {num_batches}...")
            
            # Get the current batch
            batch = np.vstack(padded_values[start_idx:end_idx])
            
            # Predict
            preds = model.predict(batch, verbose=0)
            
            # Store results
            all_preds.extend([CLASS_NAMES[np.argmax(p)] for p in preds])
            all_confidences.extend([np.max(p) for p in preds])
            
            progress_bar.progress((i + 1) / num_batches)
        
        # Add results to DataFrame
        df['Prediksi'] = all_preds
        df['Confidence'] = all_confidences
        df['Correct'] = df['Kategori'] == df['Prediksi']
        
        progress_bar.empty()
        status_text.text("Prediction completed!")
        
        return df[['Kategori', 'Konten', 'Prediksi', 'Confidence', 'Correct']]
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return pd.DataFrame()

def display_results(result_df, sample_size=10):
    # Calculate accuracy
    accuracy = result_df['Correct'].mean()
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{accuracy:.1%}")
    with col2:
        st.metric("Total Documents Processed", len(result_df))
    
    # Show a sample of results
    st.subheader(f"Sample Results (showing {min(sample_size, len(result_df))} of {len(result_df)})")
    
    # Select a balanced sample (some correct, some incorrect)
    if len(result_df) > sample_size:
        correct_samples = result_df[result_df['Correct']]
        incorrect_samples = result_df[~result_df['Correct']]
        
        # Take at least 2 incorrect samples if available
        n_incorrect = min(2, len(incorrect_samples))
        n_correct = sample_size - n_incorrect
        
        sample_df = pd.concat([
            correct_samples.sample(n=n_correct, random_state=42) if len(correct_samples) > 0 else pd.DataFrame(),
            incorrect_samples.sample(n=n_incorrect, random_state=42) if len(incorrect_samples) > 0 else pd.DataFrame()
        ])
        
        # If we didn't get enough samples, fill with random ones
        if len(sample_df) < sample_size:
            additional_samples = result_df.sample(n=sample_size-len(sample_df), random_state=42)
            sample_df = pd.concat([sample_df, additional_samples])
    else:
        sample_df = result_df
    
    # Display the sample
    def highlight_incorrect(val):
        return 'background-color: #ffcccc' if val is False else ''
    
    st.dataframe(
        sample_df.style.format({'Confidence': '{:.1%}'})
            .applymap(highlight_incorrect, subset=['Correct']),
        height=400
    )
    
    # Show misclassification analysis
    if not result_df[~result_df['Correct']].empty:
        st.subheader("Misclassification Analysis")
        
        # Confusion matrix
        st.write("Confusion Matrix:")
        confusion = pd.crosstab(
            result_df['Kategori'], 
            result_df['Prediksi'], 
            rownames=['Actual'], 
            colnames=['Predicted'],
            margins=True
        )
        st.dataframe(confusion.style.background_gradient(cmap='Blues'))
        
        # Show some misclassified examples
        st.write("Examples of Misclassified Documents:")
        wrongs = result_df[~result_df['Correct']].sample(n=min(3, len(result_df[~result_df['Correct']])), random_state=42)
        for _, row in wrongs.iterrows():
            st.write(f"**Actual:** {row['Kategori']} | **Predicted:** {row['Prediksi']} (Confidence: {row['Confidence']:.1%})")
            with st.expander("View Content"):
                st.text(row['Konten'][:500] + ("..." if len(row['Konten']) > 500 else ""))

def predict_single_text(text, model, tokenizer):
    padded = preprocess_text(text, tokenizer)
    if padded is None:
        return None, None, None
    pred = model.predict(padded, verbose=0)[0]
    return CLASS_NAMES[np.argmax(pred)], np.max(pred), pred

def bilstm_page():
    st.title("📊 Document Classification (Large Files Support)")

    # Add file upload section for model and tokenizer
    st.sidebar.header("Model Configuration")
    
    # Option to upload model files
    uploaded_model = st.sidebar.file_uploader("Upload Model (.h5 file)", type=['h5'])
    uploaded_tokenizer = st.sidebar.file_uploader("Upload Tokenizer (.json file)", type=['json'])
    
    # Save uploaded files
    if uploaded_model is not None:
        with open(MODEL_PATH, "wb") as f:
            f.write(uploaded_model.getvalue())
        st.sidebar.success("Model file uploaded successfully!")
        # Clear cache to reload model
        load_model.clear()
        
    if uploaded_tokenizer is not None:
        with open(TOKENIZER_PATH, "wb") as f:
            f.write(uploaded_tokenizer.getvalue())
        st.sidebar.success("Tokenizer file uploaded successfully!")
        # Clear cache to reload tokenizer
        load_tokenizer.clear()

    with st.spinner("Loading model and tokenizer..."):
        model = load_model()
        tokenizer = load_tokenizer()
        
        # Display error messages if any
        if model_error:
            st.error(model_error)
            st.info("Please upload a model file using the uploader in the sidebar.")
        if tokenizer_error:
            st.error(tokenizer_error)
            st.info("Please upload a tokenizer file using the uploader in the sidebar.")
            
        if model is None or tokenizer is None:
            st.warning("Model or tokenizer not loaded. Please upload the required files to continue.")
            return

    mode = st.radio("Choose Mode", ["📄 Upload File", "🔍 Single Prediction"], horizontal=True)

    if mode == "📄 Upload File":
        st.info("Note: This version supports large files (thousands of documents) with progress tracking.")
        uploaded_file = st.file_uploader("Upload CSV, Excel, or TXT file", type=['csv', 'xlsx', 'xls', 'txt'])
        
        if uploaded_file:
            # Read and process the file in chunks
            with st.spinner("Reading and processing file..."):
                df = read_data_in_chunks(uploaded_file, tokenizer)
                
            if df is not None and not df.empty:
                st.success(f"Successfully processed {len(df)} documents. Now predicting...")
                
                # Predict in batches
                result_df = predict_in_batches(df, model)
                
                if not result_df.empty:
                    # Display results
                    display_results(result_df)
                    
                    # Download options
                    st.subheader("Download Results")
                    
                    # Excel download
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        result_df.to_excel(writer, index=False, sheet_name='Results')
                    st.download_button(
                        "Download Full Results (Excel)",
                        excel_buffer.getvalue(),
                        "document_classification_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    # CSV download
                    csv_buffer = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Full Results (CSV)",
                        csv_buffer,
                        "document_classification_results.csv",
                        mime="text/csv"
                    )
    else:
        text = st.text_area("Enter text to classify:", height=200, placeholder="Paste your document content here...")
        if st.button("Classify", type="primary") and text:
            pred_class, conf, dist = predict_single_text(text, model, tokenizer)
            if pred_class:
                st.success(f"Predicted class: **{pred_class}** with confidence: **{conf:.1%}**")
                st.subheader("Prediction Distribution")
                pred_df = pd.DataFrame({"Class": CLASS_NAMES, "Confidence": dist}).sort_values("Confidence", ascending=False)
                st.bar_chart(pred_df.set_index("Class"))
                
                # Show detailed probabilities
                with st.expander("View Detailed Probabilities"):
                    st.dataframe(pred_df.style.format({'Confidence': '{:.3f}'}))

if __name__ == "__main__":
    bilstm_page()
