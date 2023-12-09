import streamlit as st
import pandas as pd
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
nltk.download('punkt')

# Function to summarize text
def summarize_text(text):
    sentences = nltk.sent_tokenize(text)
    summary = " ".join(sentences[:100])  # Take the first 100 sentences as summary
    return summary

# Function to calculate TF-IDF and cosine similarity
def calculate_similarity(summary):
    # Calculate TF-IDF
    kalimat = nltk.sent_tokenize(summary)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(kalimat)
    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    similarity_df = pd.DataFrame(cosine_similarities, columns=range(len(kalimat)), index=range(len(kalimat)))
    return similarity_df, kalimat

# Function to display graph and closeness centrality
def show_graph_centrality(similarity_df, kalimat):
    G = nx.DiGraph()
    for i in range(len(similarity_df)):
        G.add_node(i)

    for i in range(len(similarity_df)):
        for j in range(len(similarity_df)):
            similarity = similarity_df[i][j]
            if similarity > 0.02 and i != j:
                G.add_edge(i, j)

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='b')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
    nx.draw_networkx_labels(G, pos)

    plt.savefig('graph.png')
    st.image('graph.png', use_column_width=True)

    closeness_centrality = nx.closeness_centrality(G)
    sorted_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)

    st.subheader('Sorted Closeness Centrality:')
    for node, closeness in sorted_closeness:
        st.write(f"Node {node}: {closeness:.4f}")

    st.subheader('menampilkan node 3 teratas dari closeness centrality:')
    for node, closeness in sorted_closeness[:3]:
        top_sentence = kalimat[node]
        st.write(f"Node {node}: Closeness Centrality = {closeness:.4f}")
        st.write(f"Sentence: {top_sentence}\n")

# Function to calculate PageRank
def calculate_pagerank(similarity_df):
    G = nx.from_pandas_adjacency(similarity_df)
    pagerank = nx.pagerank(G)

    sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    return sorted_pagerank

# Function to display PageRank
def display_pagerank(sorted_pagerank):
    st.subheader('Sorted PageRank:')
    for node, rank in sorted_pagerank:
        st.write(f"Node {node}: {rank:.4f}")

# Function to calculate Eigenvector Centrality
def calculate_eigenvector_centrality(similarity_df):
    G = nx.from_pandas_adjacency(similarity_df)
    eigenvector = nx.eigenvector_centrality(G)

    sorted_eigenvector = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)
    return sorted_eigenvector

# Function to display Eigenvector Centrality
def display_eigenvector_centrality(sorted_eigenvector, kalimat):
    st.subheader('Sorted Eigenvector Centrality:')
    for node, eigenvector_value in sorted_eigenvector:
        st.write(f"Node {node}: {eigenvector_value:.4f}")

    st.subheader('menampilkan node 3 teratas dari eigen vektor centrality:')
    for node, eigenvector_value in sorted_eigenvector[:3]:
        top_sentence = kalimat[node]
        st.write(f"Node {node}: Eigenvector Centrality = {eigenvector_value:.4f}")
        st.write(f"Sentence: {top_sentence}\n")

# Function to calculate Eigenvalue Centrality
def calculate_eigenvalue_centrality(similarity_df):
    G = nx.from_pandas_adjacency(similarity_df)
    eigenvalue = nx.eigenvector_centrality_numpy(G)

    sorted_eigenvalue = sorted(eigenvalue.items(), key=lambda x: x[1], reverse=True)
    return sorted_eigenvalue

# Function to display Eigenvalue Centrality
def display_eigenvalue_centrality(sorted_eigenvalue, kalimat):
    st.subheader('Sorted Eigenvalue Centrality:')
    for node, eigenvalue_value in sorted_eigenvalue:
        st.write(f"Node {node}: {eigenvalue_value:.4f}")

    st.subheader('Top Nodes with Highest Eigenvalue Centrality and Respective Sentences:')
    for node, eigenvalue_value in sorted_eigenvalue[:3]:
        top_sentence = kalimat[node]
        st.write(f"Node {node}: Eigenvalue Centrality = {eigenvalue_value:.4f}")
        st.write(f"Sentence: {top_sentence}\n")

# Streamlit app
def main():
    st.title('Cosine Similarity, Graph Visualization, and Closeness Centrality')

    # Read CSV data
    data = pd.read_csv('detik_results.csv')

    # Show data in table
    st.subheader('Baca Data Detik.com')
    long_text = """
    Untuk lebih jelasnya, bisa dilihat pada link berikut: [Scraping Berita Online pada Situs Detik.com menggunakan Google Colab](https://esairina.medium.com/scraping-berita-online-pada-situs-detik-com-menggunakan-google-colab-3a764981384b)

    **Web Scraping** adalah proses pengambilan informasi dari sebuah halaman web. Ini melibatkan pengunduhan halaman web dan ekstraksi informasi dari halaman tersebut. Web scraping dapat dilakukan secara otomatis untuk mendapatkan sejumlah besar data dari berbagai situs web.
    """

    st.markdown(long_text)
    st.write(data[['headline', 'date', 'content']])

    # Select news for summarization
    selected_news_index = st.selectbox('Select news index:', data.index)
    selected_news_content = data.loc[selected_news_index, 'content']

    # Summarize selected news
    summary = summarize_text(selected_news_content)

    # Display selected news and its summary
    st.subheader('menampilka berita')
    st.write(selected_news_content)
    st.subheader('Ringkasan')
    st.write(summary)

    # Calculate similarity and display table
    st.subheader('Cosine Similarity')
    st.title('Cosine Similarity Formula')

    st.latex(r'sim(Q, D_i) = \frac{\sum_{i} WQ_j \cdot W_{i,j}}{\sqrt{\sum_{j} W^2 Q_j} \cdot \sqrt{\sum_{i} W^2_{i,j}}}')
    st.latex(r'Cs(x, y) = \frac{x \cdot y}{||x|| \cdot ||y||}')

    st.subheader('KETERANGAN:')
    st.write('- Q adalah vektor representasi dari query (pertanyaan atau pencarian) yang ingin dicocokkan dengan dokumen.')
    st.write('- D_i adalah vektor representasi dari dokumen ke-i yang ada dalam koleksi.')
    st.write('- $\sum_{i} WQ_j \cdot W_{i,j}$ adalah hasil dari perkalian titik antara vektor query $Q$ dan vektor dokumen $D_i$.')
    st.write('- $\sum_{j} WQ_j^2$ adalah penjumlahan kuadrat dari setiap elemen dalam vektor query $Q$.')
    st.write('- $\sum_{i} W_{i,j}^2$ adalah penjumlahan kuadrat dari setiap elemen dalam vektor dokumen $D_i$.')
    
    st.subheader('KETERANGAN:')
    st.write('- $X \cdot y$ = hasil kali (dot product) vektor $x$ dan $y$.')
    st.write('- $||x||$ dan $||y||$ = panjang (magnitude) dari vektor $x$ dan $y$.')
    st.write('- $||x|| \times ||y||$ = hasil kali biasa (regular product) dari dua vektor $x$ dan $y$.') 
    similarity_df, kalimat = calculate_similarity(summary)
    st.write(similarity_df)

    # Show graph and closeness centrality
    st.subheader('Graph and Closeness Centrality')
    show_graph_centrality(similarity_df, kalimat)

    # Show PageRank
    st.subheader('PageRank')
    sorted_pagerank = calculate_pagerank(similarity_df)
    display_pagerank(sorted_pagerank)

    # Display top nodes with highest PageRank and respective sentences
    st.subheader('menampilkan node 3 teratas dari page rank:')
    for node, rank in sorted_pagerank[:3]:
        top_sentence = kalimat[node]
        st.write(f"Node {node}: PageRank = {rank:.4f}")
        st.write(f"Sentence: {top_sentence}\n")

    # Show Eigenvector Centrality
    st.subheader('Eigenvector Centrality')
    sorted_eigenvector = calculate_eigenvector_centrality(similarity_df)
    display_eigenvector_centrality(sorted_eigenvector, kalimat)

    # Show Eigenvalue Centrality
    st.subheader('Eigenvalue Centrality')
    sorted_eigenvalue = calculate_eigenvalue_centrality(similarity_df)
    display_eigenvalue_centrality(sorted_eigenvalue, kalimat)

if __name__ == "__main__":
    main()
