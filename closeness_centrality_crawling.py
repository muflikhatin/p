import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import csv

# Fungsi untuk melakukan scraping detik.com
def scrape_detik(page):
    url = f'https://www.detik.com/search/searchnews?query=pemilu+2024&sortby=time&page={page}'
    req = requests.get(url)
    sop = BeautifulSoup(req.text, 'html.parser')
    li = sop.find('div', class_='list media_rows list-berita')
    lin = li.find_all('article')

    results = []
    for x in lin:
        link = x.find('a')['href']
        date = x.find('a').find('span', class_='date').text.replace('WIB','').replace('detikNews','').split(',')[1]
        headline = x.find('a').find('h2').text

        ge_ = requests.get(link).text
        sop_ = BeautifulSoup(ge_, 'html.parser')
        content = sop_.find('div', class_='detail__body-text itp_bodycontent')

        if content is not None:
            paragraphs = content.find_all('p')
            content_ = ''.join([p.get_text(strip=True) for p in paragraphs])
        else:
            content_ = ''

        results.append({
            'link': link,
            'date': date,
            'headline': headline,
            'content': content_
        })

    return results

# Fungsi untuk menambahkan hasil scraping ke dalam file CSV
def add_to_csv(results):
    with open('detik_results.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['link', 'date', 'headline', 'content'])
        writer.writerows(results)

# Fungsi utama aplikasi Streamlit
def main():
    st.title('Crawling Data Dari Detik.com')
    long_text = """
    Untuk lebih jelasnya, bisa dilihat pada link berikut: [Scraping Berita Online pada Situs Detik.com menggunakan Google Colab](https://esairina.medium.com/scraping-berita-online-pada-situs-detik-com-menggunakan-google-colab-3a764981384b)

    **Web Scraping** adalah proses pengambilan informasi dari sebuah halaman web. Ini melibatkan pengunduhan halaman web dan ekstraksi informasi dari halaman tersebut. Web scraping dapat dilakukan secara otomatis untuk mendapatkan sejumlah besar data dari berbagai situs web.
    """
    st.markdown(long_text)
    # Baca file CSV
    df = pd.read_csv('detik_results.csv', encoding='utf-8')

    # Menampilkan data di Streamlit
    st.write('mencari : ')
    st.write(df[['headline', 'date', 'content']])

    # Tambahkan tombol untuk men-trigger proses scraping
    if st.button('crawling web'):
        st.write('Scraping more data...')
        
        # Panggil fungsi scraping
        new_results = []
        for page in range(1, 1112):  # scrape 11 pages
            page_results = scrape_detik(page)
            new_results.extend(page_results)
        
        # Tambahkan hasil scraping ke dalam file CSV
        add_to_csv(new_results)

        # Tampilkan hasil terbaru
        updated_df = pd.read_csv('detik_results.csv', encoding='utf-8')
        st.write(updated_df[['headline', 'date', 'content']])



if __name__ == "__main__":
    main()


