import streamlit as st
import csv
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
import pandas as pd
import time
import random

# Set headers to mimic a browser visit
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.google.com/',
    'DNT': '1'
}

def scrape_kompas(kategori, date):
    url = f'https://indeks.kompas.com/?site={kategori.lower()}&date={date}'
    try:
        # Add random delay to avoid being blocked
        time.sleep(random.uniform(1, 3))
        
        req = requests.get(url, headers=HEADERS, verify=False, timeout=10)
        req.raise_for_status()
        
        # Check if the response contains a captcha or blocking page
        if "akses diblokir" in req.text.lower() or "captcha" in req.text.lower():
            st.warning(f"Akses diblokir untuk {url}. Silakan coba lagi nanti atau gunakan VPN.")
            return []
            
        soup = BeautifulSoup(req.text, 'html.parser')
        articles = soup.find_all('div', class_='articleItem')
        scraped_articles = []

        for article in articles:
            try:
                # Link artikel
                link = article.find('a', class_='article-link')['href']

                # Judul artikel
                title = article.find('h2', class_='articleTitle').text.strip()

                # Kategori
                category = article.find('div', class_='articlePost-subtitle').text.strip()

                # Tanggal
                tanggal = article.find('div', class_='articlePost-date').text.strip()

                # Konten artikel
                try:
                    time.sleep(random.uniform(1, 2))  # Add delay between requests
                    halaman = requests.get(link, headers=HEADERS, verify=False, timeout=10)
                    halaman.raise_for_status()
                    soup_baru = BeautifulSoup(halaman.text, 'html.parser')
                    paragraphs = soup_baru.find_all('div', class_='read__content')
                    content = ''
                    for paragraph in paragraphs:
                        for p in paragraph.find_all('p'):
                            for strong_tag in p.find_all('strong'):
                                strong_tag.replace_with('')
                            content += p.text.strip() + '\n'
                    # Hapus tanda - dari awal konten
                    content = content.lstrip('-').strip()
                except:
                    content = "Could not retrieve content"

                scraped_articles.append({
                    'Kategori': category,
                    'Judul': title,
                    'Tanggal': tanggal,
                    'Konten': content,
                    'Link': link
                })
            except Exception as e:
                st.warning(f"Error processing an article: {e}")
                continue

        return scraped_articles

    except requests.RequestException as e:
        st.error(f"Error accessing {url}: {e}")
        return []

def run_scraping():
    # Initialize session state if not exists
    if 'scraped_data' not in st.session_state:
        st.session_state.scraped_data = []
    if 'total_data' not in st.session_state:
        st.session_state.total_data = 0
        
    # Clear previous data
    st.session_state.scraped_data = []
    st.session_state.total_data = 0
    
    # Daftar kategori
    categories = ['health', 'edukasi', 'news', 'travel', 'sport']  # Using lowercase and actual kompas categories
    
    # Rentang tanggal
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 2)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_days = (end_date - start_date).days + 1
    total_tasks = total_days * len(categories)
    completed_tasks = 0
    
    current_date = start_date
    while current_date <= end_date:
        formatted_date = current_date.strftime('%Y-%m-%d')
        for category in categories:
            try:
                articles = scrape_kompas(category, formatted_date)
                if articles:  # Only extend if we got articles
                    st.session_state.scraped_data.extend(articles)
                    st.session_state.total_data += len(articles)
                
                completed_tasks += 1
                progress = completed_tasks / total_tasks
                progress_bar.progress(progress)
                status_text.text(f"Scraping... {completed_tasks}/{total_tasks} (Found {len(articles)} articles for {category} on {formatted_date})")
            except Exception as e:
                st.warning(f"Error scraping {category} on {formatted_date}: {e}")
        current_date += timedelta(days=1)
    
    progress_bar.empty()
    if st.session_state.total_data > 0:
        status_text.text(f"Scraping completed! Total articles collected: {st.session_state.total_data}")
    else:
        status_text.text("Scraping completed but no articles were collected. The website might be blocking requests.")
    
    if st.session_state.scraped_data:
        df = pd.DataFrame(st.session_state.scraped_data)
        df.to_csv('hasil_scraping.csv', index=False, encoding='utf-8')
        st.success("Data has been saved to 'hasil_scraping.csv'")

def crawling_berita_page():
    # Initialize session state variables at the start of the function
    if 'scraped_data' not in st.session_state:
        st.session_state.scraped_data = []
    if 'total_data' not in st.session_state:
        st.session_state.total_data = 0

    st.title('Kompas News Scraper')
    st.write('klik tombol dibawah ini untuk memulai scraping berita dari Kompas.com')
    
    st.warning("""
    **Note:** 
    - Website Kompas mungkin memblokir permintaan scraping.
    - Scraping mungkin tidak berhasil jika website memiliki proteksi anti-bot.
    """)

    if st.button('Start Crawling'):
        with st.spinner('Scraping in progress... Please wait (this may take several minutes)'):
            run_scraping()

    # Check if there's any data in session state
    if 'scraped_data' in st.session_state and st.session_state.scraped_data:
        st.subheader(f"First 10 of {st.session_state.total_data} Scraped Articles")
        
        df = pd.DataFrame(st.session_state.scraped_data[:3])
        
        with st.expander("View Scraped Data"):
            st.dataframe(df)
        
        st.subheader("Article Previews")
        for i, article in enumerate(st.session_state.scraped_data[:10]):
            with st.container():
                st.markdown(f"### {i+1}. {article['Judul']}")
                st.caption(f"**Category:** {article['Kategori']} | **Date:** {article['Tanggal']}")
                
                preview = article['Konten'][:200] + "..." if len(article['Konten']) > 200 else article['Konten']
                st.write(preview)
                
                st.markdown(f"[Read more]({article['Link']})")
                st.divider()
    else:
        st.info("Belum ada data. Klik tombol 'Start Crawling' untuk memulai.")
