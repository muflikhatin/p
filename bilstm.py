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
DEFAULT_MODEL_PATH = "best_model_10epochs.h5"  # Default path if no upload
TOKENIZER_PATH = "tokenizer.pkl"
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
def handle_model_upload():
    """Handle model file upload and loading with version compatibility"""
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Upload your model (best_model_10epochs.h5)",
            type=["h5"],
            accept_multiple_files=False,
            key="model_uploader"
        )
    
    if uploaded_file is not None:
        try:
            # Save the uploaded file temporarily
            temp_path = "temp_uploaded_model.h5"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Try loading with different approaches
            try:
                # First try standard loading
                model = tf.keras.models.load_model(temp_path)
                os.remove(temp_path)
                return model
            except Exception as e:
                # Fallback to custom object scope
                with tf.keras.utils.custom_object_scope({'InputLayer': tf.keras.layers.InputLayer}):
                    model = tf.keras.models.load_model(temp_path)
                os.remove(temp_path)
                return model
                
        except Exception as e:
            st.error(f"Error loading uploaded model: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None
    
    # If no upload, try loading default model
    try:
        return tf.keras.models.load_model(DEFAULT_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading default model: {str(e)}")
        return None
