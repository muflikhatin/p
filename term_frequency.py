import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import datetime as dt
import pickle
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def main():
    st.title("Memberikan Label yaitu NRP dan Komputasi")
    dt = pd.read_excel("HasilPreposPTA.xlsx")

    def get_label(Judul):
        if "RANCANG BANGUN" in str(Judul) or "SISTEM PENDUKUNG KEPUTUSAN" in str(Judul) or "IMPLEMENTASI" in str(Judul) or "APLIKASI" in str(Judul):
            return "RPL"
        else:
            return "Komputasi"
    df = pd.DataFrame(dt, columns=['Judul', 'cleaning', 'tokenized', 'stopwords_removed', 'stemmed'])

    # apply the function to the 'Judul' column and create a new 'Label' column
    df['Label'] = df['Judul'].apply(get_label)
    
    # create a dictionary to map the labels to numeric values
    label_map = {"RPL": 1, "Komputasi": 2}

    # apply the mapping to the 'Label' column
    df['Label'] = df['Label'].map(label_map)

    st.dataframe(df)
    sns.set(style="whitegrid")
    fig, ax = plt.subplots()
    sns.countplot(x='Label', data=df, ax=ax)
    st.pyplot(fig)
    
    df.to_excel("Label.xlsx", index=False)
    st.write("""
            Menampilkan jumlah Label
        """)
    # df['Label'].value_counts() 
    label_counts = df['Label'].value_counts().reset_index()
    label_counts.columns = ['Label', 'Jumlah']
    st.dataframe(label_counts)

    st.title("Halaman TF-IDF")
    st.write("""
            TF-IDF adalah singkatan dari “Term Frequency-Inverse Document Frequency” dalam pemrosesan bahasa alami dan analisis teks. Ini adalah metode statistik yang digunakan untuk mengukur pentingnya sebuah kata atau istilah dalam suatu dokumen terhadap koleksi dokumen yang lebih besar.
        """)

    # Baca data dari file Excel
    dt = pd.read_excel("HasilPreposPTA.xlsx")

    # Ekstraksi fitur dan membentuk VSM dalam term frequency
    dt['cleaning'].fillna('', inplace=True)
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(dt['cleaning'])
    count_array = count_matrix.toarray()
    feature_names = vectorizer.get_feature_names_out()  # Nama fitur
    df = pd.DataFrame(data=count_array, columns=feature_names)

    # Baca data dari file Label.xlsx
    label_df = pd.read_excel("Label.xlsx")

    # Gabungkan DataFrame df dan label_df berdasarkan indeks
    df = pd.concat([df, label_df['Label']], axis=1)

    # Simpan DataFrame ke dalam file Excel
    df.to_excel("Label_with_Features.xlsx", index=False)
    st.dataframe(df)
 
    st.title("Proses KNN")
    st.write("""
            merupakan algoritma machine learning sederhana dan mudah diterapkan yang dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi
        """)
    # Load the data
    df = pd.read_excel("Label.xlsx")
    df['cleaning'].fillna('', inplace=True)
    
    # Split the data into X and y
    X = df['cleaning']
    y = df['Label']

    # Vectorize the text data
    tfidfvectorizer = TfidfVectorizer()
    tfidf_wm = tfidfvectorizer.fit_transform(X)

    # Save the vectorizer for future use
    with open('tfidf.pkl','wb') as f:
        pickle.dump(tfidfvectorizer,f)

    # Reduce the dimensionality of the data using PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(tfidf_wm.toarray())

    # Split the data into training and testing sets
    training, test = train_test_split(X_pca,test_size=0.2, random_state=10)
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=10)

    # Train the KNN model
    modelKNN = KNeighborsClassifier(n_neighbors=3)
    modelKNN.fit(training, training_label)

    # Test the KNN model
    test_pred = modelKNN.predict(test)
    st.write(test_pred)
    accuracy = accuracy_score(test_label, test_pred)
    report = classification_report(test_label, test_pred, output_dict=True)
    
    # Print the accuracy and classification report
    st.write("KNN Model Evaluation")
    st.write("Accuracy:", accuracy)
    st.write("Classification Report:")
    st.table(report)
    # st.write("Accuracy:", accuracy_score(test_label, test_pred))
    # st.write(classification_report(test_label, test_pred))

if __name__ == "__main__":
    main()
