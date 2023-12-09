import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("K-Means Clustering")
    st.write("K-means adalah jenis metode pembelajaran tanpa pengawasan, yang digunakan ketika kita tidak memiliki data berlabel seperti dalam kasus kita, kita memiliki data tidak berlabel (berarti, tanpa kategori atau kelompok yang ditentukan). Tujuan dari algoritma ini adalah untuk menemukan kelompok dalam data, sedangkan no. kelompok diwakili oleh variabel K. Data telah dikelompokkan berdasarkan titik kesamaan yang tinggi dan titik kesamaan yang rendah dalam kelompok yang terpisah.")

    # Load the data
    df = pd.read_excel("Label_with_Features.xlsx")

    # Perform PCA
    X_std = df.values
    sklearn_pca = PCA(n_components=2)
    Y_sklearn = sklearn_pca.fit_transform(X_std)

    # Perform K-means clustering
    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, max_iter=400, algorithm='lloyd')
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

    st.title('K-means Clustering')
    st.write('Elbow Method')
    elbow_method(Y_sklearn)
    st.write('K-means Clustering')
    kmeans_clustering(Y_sklearn, fitted)
    st.write('Melihat cluster fitur teratas yang diperoleh')
    dfs = get_top_features_cluster(X_std, prediction, 20)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='score', y='features', orient='h', data=dfs[0])
    st.pyplot()

    # Silhouette Score
    silhouette_avg = silhouette_score(Y_sklearn, prediction)
    st.write(f"Silhouette Score(n=2): {silhouette_avg, Label}")

    # # Davies-Bouldin Index
    # db_score = davies_bouldin_score(Y_sklearn, prediction)
    # st.write(f"Davies-Bouldin Index: {db_score}")

    # # Calinski-Harabasz Index
    # ch_score = calinski_harabasz_score(Y_sklearn, prediction)
    # st.write(f"Calinski-Harabasz Index: {ch_score}")

if __name__ == "__main__":
    main()
