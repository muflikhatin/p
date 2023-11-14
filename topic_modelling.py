import pandas as pd
import streamlit as st
from wordcloud import WordCloud
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def main():
    st.title("Tugas 3 Latent Dirichlet Allocation (LDA)")
    st.write("""
            merupakan salah satu metode yang digunakan untuk mengelompokkan data ke dalam beberapa kelas. Penentuan pengelompokan didasarkan pada garis batas (garis lurus) yang diperoleh dari persamaan linear
        """)

    dt = pd.read_excel("HasilPreposPTA.xlsx")

    st.header("Abstrak setelah di Preprocessing")
    st.dataframe(dt['cleaning'])

    st.header("Eksplorasi Analisis (Word Cloud)")
    long_string = ','.join(list(dt['cleaning'].astype(str).values))

    wordcloud = WordCloud(background_color="white", max_words=5000,
                          contour_width=3, contour_color='steelblue')
    wordcloud.generate(long_string)

    # Visualisasi WordCloud
    st.image(wordcloud.to_image())

    st.header("Proporsi Topic Dalam Dokumen")
    data = dt['cleaning']
    # Membuat DataFrame dari data teks
    dt_lda = pd.DataFrame(data)

    dt_lda['cleaning'] = dt_lda['cleaning'].fillna('')

    # mengonversi teks menjadi matriks hitungan
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(dt_lda['cleaning'])

    # model LDA
    k = 3
    alpha = 0.1
    beta = 0.2

    lda = LatentDirichletAllocation(
        n_components=k, doc_topic_prior=alpha, topic_word_prior=beta, random_state=42)
    lda.fit(count_matrix)

    # distribusi topik pada setiap dokumen
    doc_topic_distribution = lda.transform(count_matrix)

    topic_names = [f"Topik {i+1}" for i in range(k)]
    df = pd.DataFrame(columns=['Abstrak'] + topic_names)

    for i, topic_name in enumerate(topic_names):
        df[topic_name] = doc_topic_distribution[:, i]

    # Menampilkan DataFrame
    df['Abstrak'] = dt_lda['cleaning'].values

    # Menambahkan kolom berisikan jumlah total proporsi semua topik
    df['Total Proporsi Topik'] = df[topic_names].sum(axis=1)

    # Menyimpan DataFrame sebagai file CSV
    output_csv_file = "topik_in_document.csv"
    df.to_csv(output_csv_file, index=False)

    st.dataframe(df)

    st.header("Proporsi Kata Dalam Topik")

    # Menampilkan distribusi kata pada setiap topik
    topic_word_distribution = lda.components_ / \
        lda.components_.sum(axis=1)[:, np.newaxis]

    # Membuat DataFrame untuk distribusi kata pada setiap topik
    word_topic_df = pd.DataFrame(topic_word_distribution.T, columns=[f"Topik {i+1}" for i in range(k)],
                                 index=vectorizer.get_feature_names_out())

    # Normalisasi distribusi kata pada setiap topik agar totalnya 1.0
    word_topic_df = word_topic_df.div(word_topic_df.sum(axis=1), axis=0)

    # Menambahkan kolom berisikan jumlah total proporsi semua kata pada topik
    word_topic_df['Total Proporsi Kata'] = word_topic_df.sum(axis=1)

    # Menyimpan DataFrame yang telah diperbarui sebagai file CSV
    output_csv_file = "kata_in_topik.csv"
    word_topic_df.to_csv(output_csv_file)

    st.dataframe(word_topic_df)


if __name__ == "__main__":
    main()
