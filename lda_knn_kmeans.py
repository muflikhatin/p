import pandas as pd
import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer  # Anda perlu mengimpor ini
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def main():
    st.title("Pemodelan LDA")
    st.write("""Ini adalah contoh aplikasi Streamlit untuk Latent Dirichlet Allocation (LDA).""")
    df = pd.read_excel("Label_with_Features.xlsx")
    
    # Mengubah teks menjadi fitur numerik (misalnya, TF-IDF)
    lda_results = []

    for n in range(1, 51):
        lda = LatentDirichletAllocation(n_components=n, doc_topic_prior=0.2, topic_word_prior=0.1, random_state=42, max_iter=1)
        lda_top = lda.fit_transform(df)
        lda_results.append(lda_top)
    n_components = 50
    column_names = [f'Topik {i+1}' for i in range(n_components)]
    topik50 = pd.DataFrame(lda_top, columns=column_names)
    topik50
    st.dataframe(topik50)
    
    st.title("Menambahkan kolom Label pada Dokumen Topi ")
    df_gabungan = pd.concat([topik50, df['Label']], axis=1)
    st.dataframe(df_gabungan)
    df_gabungan = df_gabungan.dropna()
    
    st.title("Klasifikasi KNN")
    X = df_gabungan.drop(columns=['Label']).values.astype('U') # Convert input data to string
    y = df_gabungan['Label'].values

    # Vectorize the text data
    tfidfvectorizer = TfidfVectorizer()
    tfidf_wm = tfidfvectorizer.fit_transform([str(x) for x in X])


    # Save the vectorizer for future use
    with open('tfidf.pkl', 'wb') as f:
        pickle.dump(tfidfvectorizer, f)

    # Reduce the dimensionality of the data using PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(tfidf_wm.toarray())

    # Split the data into training and testing sets
    training, test, training_label, test_label = train_test_split(X_pca, y, test_size=0.2, random_state=10)

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
    
    st.title("K-Means Klastering")
    X_std = df_gabungan.values
    sklearn_pca = PCA(n_components=2)
    Y_sklearn = sklearn_pca.fit_transform(X_std)

    # Perform K-means clustering
    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, max_iter=400, algorithm='auto')
    fitted = kmeans.fit(Y_sklearn)
    prediction = kmeans.predict(Y_sklearn)

    # Define the elbow method function
    def elbow_method(Y_sklearn):
        number_clusters = range(1, 7)
        kmeans = [KMeans(n_clusters=i, max_iter=600) for i in number_clusters]
        score = [kmeans[i].fit(Y_sklearn).score(Y_sklearn) for i in range(len(kmeans))]
        score = [i * -1 for i in score]
        plt.plot(number_clusters, score)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        plt.title('Elbow Method')
        st.pyplot()

    # Define the K-means clustering function
    def kmeans_clustering(Y_sklearn, fitted):
        plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=prediction, s=50, cmap='viridis')
        centers2 = fitted.cluster_centers_
        plt.scatter(centers2[:, 0], centers2[:, 1], c='black', s=300, alpha=0.6)
        st.pyplot()

    # Get top features for each cluster
    def get_top_features_cluster(X_std, prediction, n_feats):
        features = df.columns[:-1]  # Assuming the last column is the label
        labels = np.unique(prediction)
        dfs = []
        for label in labels:
            id_temp = np.where(prediction == label)
            x_means = np.mean(X_std[id_temp], axis=0)
            sorted_means = np.argsort(x_means)[::-1][:n_feats]
            best_features = [(features[i], x_means[i]) for i in sorted_means if i < len(features) and i < len(x_means)]
            Df = pd.DataFrame(best_features, columns=['features', 'score'])
            dfs.append(Df)
        return dfs

    st.write('Elbow Method')
    elbow_method(Y_sklearn)
    st.write('K-means Clustering')
    kmeans_clustering(Y_sklearn, fitted)
    # Silhouette Score
    silhouette_avg = silhouette_score(Y_sklearn, prediction)
    st.write(f"Silhouette Score: {silhouette_avg}")
    st.write('Melihat cluster fitur teratas yang diperoleh')
    dfs = get_top_features_cluster(X_std, prediction, 20)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='score', y='features', orient='h', data=dfs[0])
    st.pyplot()
    
if __name__ == "__main__":
    main()
