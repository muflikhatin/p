import time
import random

# Fungsi untuk menambahkan hasil scraping ke dalam file CSV
def add_to_csv(results):
    with open('detik_results.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['link', 'date', 'headline', 'content'])
        writer.writerows(results)

# Fungsi utama aplikasi Streamlit
def main():
    st.title('Crawling Data Dari Detik.com')
    # ...

    if st.button('crawling web'):
        st.write('Scraping more data...')

        for batch in range(1, 12):  # scrape 1 batch of 100 pages
            start_page = (batch - 1) * 100 + 1
            end_page = batch * 100 + 1

            new_results = []
            for page in range(start_page, end_page):
                try:
                    page_results = scrape_detik(page)
                    new_results.extend(page_results)
                except Exception as e:
                    st.write(f"Error scraping batch {batch}, page {page}: {e}")

                time.sleep(random.uniform(1, 3))  # Delay between requests to avoid overwhelming the server

            add_to_csv(new_results)
            time.sleep(random.uniform(5, 10))  # Pause between batches to avoid overwhelming the server
        
        updated_df = pd.read_csv('detik_results.csv', encoding='utf-8')
        st.write(updated_df[['headline', 'date', 'content']])
