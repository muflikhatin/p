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
    st.write(f"TensorFlow: {tf.__version__}")
    st.write(f"Keras: {tf.keras.__version__}")
    st.write(f"Recommended: TensorFlow {RECOMMENDED_TF_VERSION}")

def load_tokenizer(path=TOKENIZER_PATH):
    """Load the tokenizer with proper error handling"""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Tokenizer file not found at: {os.path.abspath(path)}")
        st.error("Please ensure the tokenizer.pkl exists in your directory")
    except Exception as e:
        st.error(f"Error loading tokenizer: {str(e)}")
    return None

def load_model_with_fallback(uploaded_file):
    """Attempt to load model from uploaded file with version compatibility fallbacks"""
    try:
        # Save uploaded file to a temporary location
        temp_model_path = "temp_model.h5"
        with open(temp_model_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Load the model
        model = tf.keras.models.load_model(temp_model_path)
        st.success("Model loaded successfully from uploaded file!")
        
        # Clean up temporary file
        os.remove(temp_model_path)
        
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        
        # Show detailed troubleshooting
        with st.expander("Version Compatibility Solutions"):
            st.markdown(f"""
            ### Detected TensorFlow {tf.__version__} but model might require ~{RECOMMENDED_TF_VERSION}

            1. **Recommended**: Create fresh environment:
            ```bash
            python -m venv tf_env
            source tf_env/bin/activate  # Linux/Mac
            .\\tf_env\\Scripts\\activate  # Windows
            pip install tensorflow=={RECOMMENDED_TF_VERSION}
            ```

            2. **Model Conversion** (if you have original):
            ```python
            import tensorflow as tf
            model = tf.keras.models.load_model('original_model.h5')
            model.save('converted_model.h5')
            ```

            3. **Last Resort**: Retrain model with current TF version
            """)
            display_versions()
        
        return None

def preprocess_text(text, tokenizer):
    """Convert raw text to padded sequences"""
    sequences = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

def display_prediction_results(predictions):
    """Display classification results in user-friendly format"""
    pred_class = np.argmax(predictions)
    confidence = predictions[pred_class]
    
    # Main result
    st.success(f"**Predicted Category**: {CLASS_NAMES[pred_class]} (confidence: {confidence:.1%})")
    
    # Detailed probabilities
    st.subheader("Category Probabilities")
    prob_data = pd.DataFrame({
        'Category': CLASS_NAMES,
        'Probability': predictions,
        'Confidence (%)': (predictions * 100).round(1)
    }).sort_values('Probability', ascending=False)
    
    # Add visual bar chart
    st.bar_chart(prob_data.set_index('Category')['Probability'])
    st.table(prob_data)

def bilstm_page():
    """Main BiLSTM classification interface"""
    st.title("📄 Document Classification with BiLSTM")
    
    # Version info (collapsed by default)
    with st.expander("Environment Information", expanded=False):
        display_versions()
    
    # Model loading section
    st.subheader("Model Configuration")
    
    # File uploader for model
    uploaded_model = st.file_uploader(
        "Upload BiLSTM Model (.h5 file)",
        type=["h5"],
        help="Upload your trained BiLSTM model in .h5 format"
    )
    
    with st.spinner('Loading NLP resources...'):
        col1, col2 = st.columns(2)
        with col1:
            model = load_model_with_fallback(uploaded_model) if uploaded_model else None
            if not uploaded_model:
                st.warning("Please upload a model file to proceed")
        with col2:
            tokenizer = load_tokenizer()
        
        if not model or not tokenizer:
            st.error("⚠️ System cannot proceed without both model and tokenizer")
            st.stop()  # Stop execution if resources not loaded
    
    # Classification interface
    st.subheader("Text Classification")
    text_input = st.text_area(
        "Enter news/article text:", 
        height=200,
        placeholder="Paste your content here...",
        help="Input text to classify into one of the predefined categories"
    )
    
    if st.button("Classify Text"):
        if not text_input.strip():
            st.warning("Please input text to classify")
            return
            
        with st.spinner('Analyzing content...'):
            try:
                # Text processing
                padded_seq = preprocess_text(text_input, tokenizer)
                
                # Prediction
                predictions = model.predict(padded_seq, verbose=0)[0]
                
                # Display results
                display_prediction_results(predictions)
                
            except Exception as e:
                st.error(f"Classification failed: {str(e)}")
                st.error("Please check your input and try again")

if __name__ == "__main__":
    bilstm_page()
