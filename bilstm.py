import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from io import BytesIO

# Constants
CLASS_NAMES = ['Travel', 'Edukasi', 'Sports', 'Politik', 'Health']
MAX_SEQUENCE_LENGTH = 300
MODEL_PATH = "best_model_10epochs.h5"
TOKENIZER_GITHUB_RAW_URL = "https://github.com/muflikhatin/p/blob/main/tokenizer.pkl"
RECOMMENDED_TF_VERSION = "2.6.0"

def display_versions():
    """Display version information for troubleshooting"""
    st.write(f"TensorFlow: {tf.__version__}")
    try:
        keras_version = getattr(tf.keras, '__version__', 
                              getattr(tf.keras, 'version', 
                                     "Built-in with TensorFlow"))
        st.write(f"Keras: {keras_version}")
    except Exception:
        st.write("Keras: Built-in with TensorFlow")
    st.write(f"Recommended: TensorFlow {RECOMMENDED_TF_VERSION}")

def download_tokenizer_from_github():
    """Download tokenizer directly from GitHub without saving locally"""
    try:
        response = requests.get(TOKENIZER_GITHUB_RAW_URL)
        response.raise_for_status()
        tokenizer = pickle.load(BytesIO(response.content))
        st.success("Tokenizer loaded successfully from GitHub!")
        return tokenizer
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download tokenizer from GitHub: {str(e)}")
    except pickle.PickleError as e:
        st.error(f"Error unpickling tokenizer: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error loading tokenizer: {str(e)}")
    return None

def load_model_with_fallback():
    """Load model with version compatibility handling"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        with st.expander("Troubleshooting Guide"):
            st.markdown(f"""
            ### Compatibility Issues Detected
            **Recommended Solutions:**
            1. **Install Recommended Version**:
            ```bash
            pip install tensorflow=={RECOMMENDED_TF_VERSION}
            ```
            2. **Convert Model**:
            ```python
            import tensorflow as tf
            model = tf.keras.models.load_model('{MODEL_PATH}')
            model.save('converted_model.h5', save_format='h5')
            ```
            3. **Environment Setup**:
            ```bash
            python -m venv tf_env
            source tf_env/bin/activate  # Linux/Mac
            .\\tf_env\\Scripts\\activate  # Windows
            pip install tensorflow=={RECOMMENDED_TF_VERSION}
            ```
            """)
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
    st.subheader("Detailed Probabilities")
    prob_data = pd.DataFrame({
        'Category': CLASS_NAMES,
        'Probability': predictions,
        'Confidence (%)': (predictions * 100).round(1)
    }).sort_values('Probability', ascending=False)
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(prob_data.set_index('Category')['Probability'])
    with col2:
        st.table(prob_data)

def bilstm_page():
    """Main BiLSTM classification interface - renamed from main_app_interface"""
    st.title("📰 News Article Classifier")
    st.markdown("Classify articles into: Travel, Edukasi, Sports, Politik, or Health")
    
    with st.expander("System Information", expanded=False):
        display_versions()
    
    # Load resources
    with st.spinner("Loading NLP components..."):
        tokenizer = download_tokenizer_from_github()
        model = load_model_with_fallback()
        
        if not tokenizer or not model:
            st.error("Critical components failed to load. Cannot proceed.")
            st.stop()
    
    # User input
    st.subheader("Input Text for Classification")
    user_text = st.text_area(
        "Paste your article content here:",
        height=250,
        placeholder="Enter news/article text to classify...",
        help="Minimum 50 characters for reliable classification"
    )
    
    if st.button("Classify Article"):
        if not user_text or len(user_text.strip()) < 50:
            st.warning("Please enter sufficient text for classification (min 50 chars)")
            return
            
        with st.spinner("Analyzing content..."):
            try:
                processed = preprocess_text(user_text, tokenizer)
                predictions = model.predict(processed, verbose=0)[0]
                display_prediction_results(predictions)
            except Exception as e:
                st.error(f"Classification error: {str(e)}")
                st.error("Please try again with different text")

if __name__ == "__main__":
    st.set_page_config(
        page_title="News Classifier",
        page_icon="📰",
        layout="wide"
    )
    bilstm_page()
