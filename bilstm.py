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
TOKENIZER_PATH = "tokenizer.pkl"
RECOMMENDED_TF_VERSION = "2.6.0"

def display_versions():
    """Display version information for troubleshooting"""
    st.write(f"TensorFlow Version: {tf.__version__}")
    st.write(f"Recommended Version: {RECOMMENDED_TF_VERSION}")

def load_tokenizer(path=TOKENIZER_PATH):
    """Load the tokenizer with proper error handling"""
    try:
        with open(path, 'rb') as f:
            tokenizer = pickle.load(f)
            
        # Verify tokenizer functionality
        if hasattr(tokenizer, 'texts_to_sequences'):
            test_text = "sample text"
            sequences = tokenizer.texts_to_sequences([test_text])
            if sequences:
                return tokenizer
        return tokenizer
        
    except FileNotFoundError:
        st.error(f"Tokenizer file not found at: {os.path.abspath(path)}")
    except Exception as e:
        st.error(f"Error loading tokenizer: {str(e)}")
        st.error("Possible version mismatch. Try:")
        st.error("1. Recreate tokenizer with current TF version")
        st.error(f"2. Use TensorFlow {RECOMMENDED_TF_VERSION}")
    return None

def load_model_with_fallback(uploaded_file):
    """Attempt to load model from uploaded file with version compatibility fallbacks"""
    try:
        # Save uploaded file to a temporary location
        temp_model_path = "temp_model.h5"
        with open(temp_model_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Try different loading methods
        try:
            model = tf.keras.models.load_model(temp_model_path)
        except:
            try:
                model = tf.keras.models.load_model(temp_model_path, compile=False)
            except Exception as e:
                st.error(f"Advanced loading failed: {str(e)}")
                raise
                
        st.success("Model loaded successfully!")
        os.remove(temp_model_path)
        return model
        
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        
        with st.expander("Troubleshooting Guide"):
            st.markdown(f"""
            ### Version Compatibility Solutions
            
            **Current TensorFlow**: {tf.__version__}  
            **Recommended Version**: {RECOMMENDED_TF_VERSION}

            1. **Create compatible environment**:
            ```bash
            python -m venv tf_env
            source tf_env/bin/activate  # Linux/Mac
            .\\tf_env\\Scripts\\activate  # Windows
            pip install tensorflow=={RECOMMENDED_TF_VERSION}
            ```

            2. **Model conversion**:
            ```python
            import tensorflow as tf
            model = tf.keras.models.load_model('original.h5', compile=False)
            model.save('converted.h5', save_format='h5')
            ```

            3. **Retrain model** with current TF version
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
    
    with st.expander("Environment Info", expanded=False):
        display_versions()
    
    st.subheader("1. Model Configuration")
    uploaded_model = st.file_uploader(
        "Upload BiLSTM Model (.h5)",
        type=["h5"],
        help="Upload your trained model file"
    )
    
    with st.spinner('Loading resources...'):
        col1, col2 = st.columns(2)
        with col1:
            model = load_model_with_fallback(uploaded_model) if uploaded_model else None
            if not uploaded_model:
                st.warning("Please upload a model file")
        with col2:
            tokenizer = load_tokenizer()
        
        if not model or not tokenizer:
            st.error("⚠️ System initialization failed")
            st.stop()
    
    st.subheader("2. Text Classification")
    text_input = st.text_area(
        "Enter text to classify:", 
        height=200,
        placeholder="Paste content here..."
    )
    
    if st.button("Classify Text"):
        if not text_input.strip():
            st.warning("Please input text")
            return
            
        with st.spinner('Analyzing...'):
            try:
                padded_seq = preprocess_text(text_input, tokenizer)
                predictions = model.predict(padded_seq, verbose=0)[0]
                display_prediction_results(predictions)
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    bilstm_page()
