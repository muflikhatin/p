import nltk
import pandas as pd
import streamlit as st
import re, string
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from itertools import chain
from indoNLP.preprocessing import pipeline, replace_word_elongation, replace_slang, emoji_to_words, remove_html
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from scipy.sparse import hstack
import joblib

def main():
    # Download resources
    nltk.download('vader_lexicon')
    nltk.download('stopwords')

    # Load dataset
    df = pd.read_csv('categories.csv')
    df 
    df.drop_duplicates(inplace=True)
    df = df.drop(['date', 'headline'], axis=1)

    # Text Cleaning
    def cleaning(text):
        text = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});').sub('', str(text))
        text = text.lower()
        text = text.strip()
        text = re.compile('<.*?>').sub('', text)
        text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
        text = re.sub('\s+', ' ', text)
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
        text = re.sub(r'\d', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub('nan', '', text)
        return text

    def preprocess_data(df):
        # Preprocess text
        pipe = pipeline([replace_word_elongation, replace_slang, emoji_to_words, remove_html])
        df['content(clean)'] = df['content'].astype(str).apply(lambda x: pipe(x))
        df['content(clean)'] = df['content(clean)'].apply(lambda x: cleaning(x))
        df['content(clean)'] = df['content(clean)'].replace('', np.nan)
        df.dropna(subset=['content(clean)'], inplace=True)
        df = df[['content', 'content(clean)', 'category']]
        
        # Map string labels to integers
        label_mapping = {'pemilu+2024': 0, 'sport': 1, 'edu': 2, 'kategori_tambahan': 3}  # Ganti dengan label kategori Anda
        df['category'] = df['category'].map(label_mapping)
        df.dropna(subset=['category'], inplace=True)  # Hapus baris dengan kategori NaN

        return df
        # Tokenizing tweets
        tknzr = TweetTokenizer()
        df['token'] = df['content(clean)'].apply(lambda x: tknzr.tokenize(x))
        
        # Removing stopwords
        stop_words = set(chain(stopwords.words('indonesian'), stopwords.words('english')))
        df['token'] = df['token'].apply(lambda x: [w for w in x if not w in stop_words])
        return df
    
        # Initialize tokenizer
       

        # Initialize preprocessing pipeline
        pipe = pipeline([replace_word_elongation, replace_slang, emoji_to_words, remove_html])

    # Preprocess data
    @st.cache()
    def preprocess_data_cached(df):
        return preprocess_data(df)
    def select_features(data, labels, num_features=2000):
        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(data)
        
        # Calculate information gain scores
        info_gain = mutual_info_classif(features, labels)
        
        # Select top k features with highest scores
        selector = SelectKBest(mutual_info_classif, k=num_features)
        selected_features = selector.fit_transform(features, labels)
        
        # Get indices of selected features
        selected_indices = selector.get_support(indices=True)
        
        # Get feature names
        feature_names = np.array(vectorizer.get_feature_names_out())[selected_indices]
        
        return selected_features, feature_names



    # Streamlit App
    st.title("Klasifikasi dan Pemodelan")
    st.write(" ")
    st.header("Dataset Sebelum Dilakukan Preprocessing")
    st.dataframe(df.dropna(subset=['category']))  # Display DataFrame after dropping NaN values in 'category' column

    # Rest of your Streamlit code...
    # ... (Sentiment analysis, Model evaluation, etc.)
    

    st.header("Data Hasil Preprocessing")
    if st.button("Preprocessing Data"):
        processed_data = preprocess_data_cached(df)
        st.success("Preprocessing data selesai.")
        st.dataframe(processed_data)

    # ... (Bagian kode sebelumnya tetap sama)

    st.header("Klasifikasi SVM")
    text_to_classify = st.text_input("Masukkan teks untuk klasifikasi:")
    if text_to_classify:
        # Preprocess the input text
        processed_text = cleaning(text_to_classify)  # Preprocessing function
        
        # Load the trained model
        svm_model = joblib.load('svm_model_with_info_gain.pkl')
        
        # Load the vectorizer used during training
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        
        # Transform the processed text into feature vectors using the loaded vectorizer
        text_vector = vectorizer.transform([processed_text])
        
        # Ensure that the text vector has the same dimensions as expected by the model
        if text_vector.shape[1] != 2000:
            # Perform feature selection or reduction to ensure the vector has 1000 features
            selector = SelectKBest(mutual_info_classif, k=2000)
            selected_features = selector.fit_transform(text_vector, [0])  # Assuming 0 as the label for the processed_text
            text_vector = selected_features
        
        # Predict the category
        category_prediction = svm_model.predict(text_vector)
        st.success(f"Kategori Prediksi: {category_prediction[0]}")


    if st.button("Evaluasi Model"):
    # Splitting the data
        processed_data = preprocess_data(df)

        # Separate texts and labels
        texts = processed_data['content(clean)'].values
        labels = processed_data['category'].values

        # Extract features using TF-IDF
        vectorizer = TfidfVectorizer()
        features_tfidf = vectorizer.fit_transform(texts)

        # Feature selection using Information Gain
        selector = SelectKBest(mutual_info_classif, k=2000)
        selected_features = selector.fit_transform(features_tfidf, labels)

        # Split dataset into training and testing data
        features_train, features_test, labels_train, labels_test = train_test_split(selected_features, labels, test_size=0.2, random_state=42)

        # Build SVM model
        svm_model = SVC(kernel='linear')
        svm_model.fit(features_train, labels_train)

        # Predict with SVM model
        predictions = svm_model.predict(features_test)

        # Calculate accuracy, precision, and recall
        accuracy = accuracy_score(labels_test, predictions)
        precision = precision_score(labels_test, predictions, average='weighted')
        recall = recall_score(labels_test, predictions, average='weighted')

        # Display evaluation results
        st.write("Akurasi:", accuracy)
        st.write("Presisi:", precision)
        st.write("Recall:", recall)

        # Save the trained SVM model
        joblib.dump(svm_model, 'svm_model_with_info_gain.pkl')

if __name__ == "__main__":
    main()
