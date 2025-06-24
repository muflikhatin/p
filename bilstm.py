import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
CLASS_NAMES = ['Travel', 'Edukasi', 'Sports', 'Politik', 'Health']
MAX_SEQUENCE_LENGTH = 300
RECOMMENDED_TF_VERSION = "2.6.0"

def display_versions():
    """Display version information for troubleshooting"""
    st.write(f"TensorFlow: {tf.__version__}")
    try:
        st.write(f"Keras: {tf.keras.__version__}")
    except AttributeError:
        st.write("Keras: Built-in with TensorFlow")
    st.write(f"Recommended: TensorFlow {RECOMMENDED_TF_VERSION}")

def handle_file_upload(file_type, file_ext):
    """Handle file uploads with proper validation"""
    uploaded_file = st.file_uploader(
        f"Upload {file_type} file (.{file_ext})",
        type=[file_ext],
        key=f"{file_type}_uploader"
    )
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_path = f"temp_uploaded.{file_ext}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return temp_path
    return None

def load_tokenizer(uploaded_path=None):
    """Load tokenizer from uploaded file or default path"""
    if uploaded_path:
        try:
            with open(uploaded_path, 'rb') as f:
                tokenizer = pickle.load(f)
            os.remove(uploaded_path)  # Clean up temp file
            return tokenizer
        except Exception as e:
            st.error(f"Error loading tokenizer: {str(e)}")
            return None
    
    st.info("No tokenizer uploaded yet")
    return None

def load_model(uploaded_path=None):
    """Load model with comprehensive error handling"""
    if uploaded_path:
        try:
            # Try standard loading first
            try:
                model = tf.keras.models.load_model(uploaded_path)
                st.success("Model loaded successfully!")
                os.remove(uploaded_path)  # Clean up temp file
                return model
            except Exception as e:
                st.warning(f"Standard loading failed: {str(e)}. Trying compatibility mode...")
            
            # Try with custom objects
            try:
                custom_objects = {
                    'InputLayer': tf.keras.layers.InputLayer,
                    'Functional': tf.keras.models.Model
                }
                with tf.keras.utils.custom_object_scope(custom_objects):
                    model = tf.keras.models.load_model(uploaded_path)
                st.success("Model loaded in compatibility mode!")
                os.remove(uploaded_path)
                return model
            except Exception as e:
                st.error(f"Model loading failed: {str(e)}")
                return None
        except Exception as e:
            st.error(f"Error processing model file: {str(e)}")
            return None
    
    st.info("No model uploaded yet")
    return None

def preprocess_text(text, tokenizer):
    """Convert raw text to padded sequences"""
    sequences = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

def display_prediction_results(predictions):
    """Display classification results"""
    pred_class = np.argmax(predictions)
    confidence = predictions[pred_class]
    
    st.success(f"**Predicted Category**: {CLASS_NAMES[pred_class]} (confidence: {confidence:.1%})")
    
    st.subheader("Category Probabilities")
    prob_data = pd.DataFrame({
        'Category': CLASS_NAMES,
        'Probability': predictions,
        'Confidence (%)': (predictions * 100).round(1)
    }).sort_values('Probability', ascending=False)
    
    st.bar_chart(prob_data.set_index('Category')['Probability'])
    st.table(prob_data)

def bilstm_page():
    """Main BiLSTM classification interface"""
    st.title("📄 Document Classification with BiLSTM")
    
    with st.expander("Environment Information", expanded=False):
        display_versions()
    
    st.subheader("1. Upload Required Files")
    
    # File upload section
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Upload Model File**")
        model_path = handle_file_upload("model", "h5")
    with col2:
        st.markdown("**Upload Tokenizer File**")  
        tokenizer_path = handle_file_upload("tokenizer", "pkl")

    # Load resources
    tokenizer = load_tokenizer(tokenizer_path)
    model = load_model(model_path)

    st.subheader("2. Enter Text for Classification")
    input_text = st.text_area("Paste your document text here:", height=200)

    if st.button("Classify Document") and tokenizer and model:
        if input_text.strip():
            with st.spinner("Processing document..."):
                try:
                    # Preprocess and predict
                    processed_text = preprocess_text(input_text, tokenizer)
                    predictions = model.predict(processed_text)
                    display_prediction_results(predictions[0])
                except Exception as e:
                    st.error(f"Error during classification: {str(e)}")
        else:
            st.warning("Please enter some text to classify")

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Document Classifier",
        page_icon="📄",
        layout="wide"
    )

    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose the page:",
        ["BiLSTM Classifier", "About"]
    )

    if app_mode == "BiLSTM Classifier":
        bilstm_page()
    else:
        st.title("About This App")
        st.markdown("""
        ## Document Classification Tool
        
        This application uses a BiLSTM neural network to classify documents into 5 categories:
        - Travel
        - Edukasi (Education)
        - Sports  
        - Politik (Politics)
        - Health

        ### How to use:
        1. Upload a trained BiLSTM model (.h5 file)
        2. Upload the corresponding tokenizer (.pkl file)
        3. Enter text to classify
        4. Click "Classify Document"

        ### Requirements:
        - TensorFlow 2.x (Recommended: 2.6.0)
        - Python 3.7+
        """)

if __name__ == "__main__":
    main()
