import streamlit as st
import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd


def main():
    st.markdown("")
    st.title("Crawling Website https://www.kompas.com/ with Beautiful Soap")
    st.write("""
        Crawling data adalah proses otomatis untuk mengumpulkan dan mengindeks data dari berbagai sumber seperti situs web, database, atau dokumen.
        Crawling data dilakukan pada website https://www.kompas.com/
        """)

    def run_crawling_process():
        try:
            prod = '10'
            page = 1

            # Membuat file CSV untuk menulis hasil scraping
            with open('df.csv', 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Kategori', 'Judul', 'Tanggal',
                              'Junlah_Kata', 'Konten', 'Link']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                while True:
                    base_url = 'https://pta.trunojoyo.ac.id/c_search/byprod/{}/{}'.format(
                        prod, page)
                    url = base_url
                    req = requests.get(url)
                    soup = BeautifulSoup(req.text, 'html.parser')
                    items = soup.find_all('li', attrs={'data-id': 'id-1'})

                    if not items:
                        # Jika tidak ada item di halaman saat ini, keluar dari loop
                        break

                    for it in items:
                        data = {}
                        title = it.find('a', class_='title').text
                        data['Judul'] = title
                        div_elements = it.find_all(
                            'div', style='padding:2px 2px 2px 2px;')
                        for div in div_elements:
                            span = div.find('span')
                            if span:
                                span_text = span.get_text()
                                key, value = span_text.split(':', 1)
                                data[key.strip()] = value.strip()

                        abstrak_button = it.find('a', class_='gray button')
                        if abstrak_button:
                            abstrak_link = abstrak_button['href']
                            abstrak_req = requests.get(abstrak_link)
                            abstrak_soup = BeautifulSoup(
                                abstrak_req.text, 'html.parser')
                            abstrak = abstrak_soup.find('p', align='justify')
                            if abstrak:
                                abstrak_text = abstrak.get_text(strip=True)
                                data['Abstrak'] = abstrak_text
                            else:
                                data['Abstrak'] = "Tidak ditemukan abstrak"

                        writer.writerow(data)
                        st.text("Data berhasil ditambahkan: {}".format(data))

                    page += 1

            st.success("Proses crawling selesai.")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

    # Tampilkan UI Streamlit
    st.header("proses crowling data")

    if st.button("Mulai Web crowling"):
        st.text("Memulai proses crawling...")
        run_crawling_process()

    abst = pd.read_csv("hasil_crowling.csv")
    st.dataframe(abst)


if __name__ == "__main__":
    main()
