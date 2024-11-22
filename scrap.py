import os
import csv
import time
import random
import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import quote
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deepseek import DeepSeek

class WebScraper:
    def __init__(self):
        self.search_engines = [
            {
                'name': 'Google',
                'url': 'https://www.google.com/search?q=',
                'result_selector': 'div.g div.yuRUbf a',
                'title_selector': 'div.g h3',
                'snippet_selector': 'div.g div.VwiC3b'
            },
            {
                'name': 'Bing',
                'url': 'https://www.bing.com/search?q=',
                'result_selector': '#b_results h2 a',
                'title_selector': '#b_results h2',
                'snippet_selector': '#b_results .b_caption p'
            },
            {
                'name': 'DuckDuckGo',
                'url': 'https://html.duckduckgo.com/html/?q=',
                'result_selector': 'a.result__a',
                'title_selector': 'a.result__a',
                'snippet_selector': 'a.result__snippet'
            },
            {
                'name': 'Qwant',
                'url': 'https://www.qwant.com/?q=',
                'result_selector': '.result__url',
                'title_selector': '.result__title',
                'snippet_selector': '.result__desc'
            },
            {
                'name': 'Ecosia',
                'url': 'https://www.ecosia.org/search?q=',
                'result_selector': '.result__link',
                'title_selector': '.result__title',
                'snippet_selector': '.result__snippet'
            },
            {
                'name': 'Startpage',
                'url': 'https://www.startpage.com/do/search?q=',
                'result_selector': '.w-gl__result-url',
                'title_selector': '.w-gl__result-title',
                'snippet_selector': '.w-gl__description'
            }
        ]
        self.ua = UserAgent()
        self.output_dir = 'try_search'
        self.create_output_directory()
        self.setup_selenium()
        self.setup_sentiment_analyzers()

        # Initialize DeepSeek
        try:
            self.deepseek = DeepSeek()
        except Exception as e:
            print(f"[WARNING] Failed to initialize DeepSeek: {str(e)}")
            self.deepseek = None
    
    def get_content_summary(self, content):
        if self.deepseek is None:
            return content
            
        try:
            summary = self.deepseek.summary_web(content, max_tokens=4000) #max token
            return summary
        except Exception as e:
            print(f"[WARNING] Summary generation failed: {str(e)}")
            return content
        
    

    def setup_sentiment_analyzers(self):
        print("[INFO] Setting up sentiment analyzers")
        try:
            nltk.download('punkt', quiet=True)
        except:
            print("[WARNING] NLTK punkt download failed, but continuing...")
        self.vader = SentimentIntensityAnalyzer()

    def perform_sentiment_analysis(self, text):
        try:
            # Split text into sentences
            sentences = sent_tokenize(text)
            
            # TextBlob analysis
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # VADER analysis
            vader_scores = self.vader.polarity_scores(text)
            
            # Calculate per-sentence sentiment
            sentence_sentiments = []
            for sentence in sentences[:]:  # Limit to first ? sentences
                vader_sent_scores = self.vader.polarity_scores(sentence)
                blob_sent = TextBlob(sentence)
                
                sentence_sentiment = {
                    'text': sentence[:],  # Limit ? sentence length
                    'vader': vader_sent_scores,
                    'textblob': {
                        'polarity': blob_sent.sentiment.polarity,
                        'subjectivity': blob_sent.sentiment.subjectivity
                    }
                }
                sentence_sentiments.append(sentence_sentiment)

            # Compile results
            sentiment_results = {
                'overall': {
                    'vader': vader_scores,
                    'textblob': {
                        'polarity': textblob_polarity,
                        'subjectivity': textblob_subjectivity
                    }
                },
                'sentence_analysis': sentence_sentiments,
                'summary': {
                    'avg_sentiment': (vader_scores['compound'] + textblob_polarity) / 2,
                    'sentiment_category': self.get_sentiment_category(vader_scores['compound'], textblob_polarity)
                }
            }
            
            return json.dumps(sentiment_results)
            
        except Exception as e:
            print(f"[ERROR] Sentiment analysis failed: {str(e)}")
            return json.dumps({'error': str(e)})
        
    def perform_overall_sentiment_analysis(self, text):
        try:
            # TextBlob analysis
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # VADER analysis
            vader_scores = self.vader.polarity_scores(text)
            
            # Create concise sentiment summary
            sentiment_summary = {
                'compound_score': round(vader_scores['compound'], 3),
                'polarity': round(textblob_polarity, 3),
                'subjectivity': round(textblob_subjectivity, 3),
                'sentiment_category': self.get_sentiment_category(vader_scores['compound'], textblob_polarity),
                'sentiment_distribution': {
                    'positive': round(vader_scores['pos'], 3),
                    'neutral': round(vader_scores['neu'], 3),
                    'negative': round(vader_scores['neg'], 3)
                }
            }
            
            return json.dumps(sentiment_summary)
                
        except Exception as e:
            print(f"[ERROR] Sentiment analysis failed: {str(e)}")
            return json.dumps({'error': str(e)})

    def get_sentiment_category(self, vader_compound, textblob_polarity):
        avg_score = (vader_compound + textblob_polarity) / 2
        if avg_score >= 0.3:
            return 'very_positive'
        elif avg_score >= 0.1:
            return 'positive'
        elif avg_score <= -0.3:
            return 'very_negative'
        elif avg_score <= -0.1:
            return 'negative'
        else:
            return 'neutral'

    def setup_selenium(self):
        print("[INFO] Setting up Selenium WebDriver")
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument(f'user-agent={self.ua.random}')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--disable-notifications')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": self.ua.random})

    def create_output_directory(self):
        print("[INFO] Creating output directory")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def fetch_with_selenium(self, url, retry_count=2):
        for attempt in range(retry_count):
            try:
                print(f"[INFO] Fetching URL (attempt {attempt + 1}): {url}")
                self.driver.get(url)
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                time.sleep(random.uniform(3, 5))
                return self.driver.page_source
            except Exception as e:
                print(f"[ERROR] Attempt {attempt + 1} failed: {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(random.uniform(2, 4))
                    continue
        return None

    def append_to_csv(self, keyword, result):
        filename = os.path.join(self.output_dir, f"{keyword.replace(' ', '_')}.csv")
        file_exists = os.path.exists(filename)
        
        with open(filename, 'a', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL, escapechar='\\')
            if not file_exists:
                writer.writerow(['Search Engine', 'URL', 'Content', 'Sentiment Analysis', 'Summary', 'Summary Sentiment'])
            writer.writerow([
                result['engine'],
                result['url'],
                result['content'],
                result['sentiment'],
                result['summary'],
                result['summary_sentiment']
            ])
        print(f"[INFO] Successfully appended result to CSV: {filename}")

    def search_and_scrape(self, keywords=["starting a business", "selling  in 2024"]):
        self.keywords = keywords
        file_locations = []
        aggregated_data = []
        try:
            for keyword in self.keywords:
                print(f"\n[INFO] Processing keyword: {keyword}")
                results_count = 0
                keyword_urls = []
                keyword_summaries = []
                
                # Get initial row count from existing file
                filename = f"try_search/{keyword.replace(' ', '_')}.csv"
                if os.path.exists(filename):
                    with open(filename, 'r', encoding='utf-8-sig') as csvfile:
                        row_number = sum(1 for row in csv.reader(csvfile))
                else:
                    row_number = 1

                for engine in self.search_engines:
                    if results_count >= 3:
                        break

                    print(f"\n[INFO] Trying search engine: {engine['name']}")
                    try:
                        search_url = f"{engine['url']}{quote(keyword)}"
                        page_source = self.fetch_with_selenium(search_url)
                        
                        if not page_source:
                            continue

                        soup = BeautifulSoup(page_source, 'html.parser')
                        links = soup.select(engine['result_selector'])
                        
                        if not links:
                            continue

                        print(f"[INFO] Found {len(links)} potential results on {engine['name']}")
                        
                        for link in links:
                            if results_count >= 3:
                                break
                                
                            href = link.get('href')
                            if href and href.startswith('http'):
                                try:
                                    print(f"[INFO] Fetching content from: {href}")
                                    page_content = self.fetch_with_selenium(href)
                                    
                                    if page_content:
                                        page_soup = BeautifulSoup(page_content, 'html.parser')
                                        content = ' '.join(page_soup.get_text(strip=True).split())
                                        
                                        print("[INFO] Performing sentiment analysis")
                                        sentiment_results = self.perform_sentiment_analysis(content)
                                        
                                        print("[INFO] Generating summary")
                                        content_summary = self.get_content_summary(content)
                                        
                                        print("[INFO] Performing sentiment analysis on summary")
                                        summary_sentiment_results = self.perform_sentiment_analysis(content_summary)
                                        
                                        result = {
                                            'url': href,
                                            'content': content,
                                            'engine': engine['name'],
                                            'sentiment': sentiment_results,
                                            'summary': content_summary,
                                            'summary_sentiment': summary_sentiment_results
                                        }
                                        
                                        self.append_to_csv(keyword, result)
                                        filename = f"try_search/{keyword.replace(' ', '_')}.csv"
                                        file_locations.append(f"{filename}[{row_number}]")
                                        
                                        keyword_urls.append(href)
                                        keyword_summaries.append(content_summary)
                                        
                                        results_count += 1
                                        row_number += 1
                                    
                                except Exception as e:
                                    print(f"[ERROR] Failed to fetch page content: {str(e)}")
                            
                            time.sleep(random.uniform(4, 7))
                    
                    except Exception as e:
                        print(f"[ERROR] {engine['name']} search failed: {str(e)}")
                    
                    time.sleep(random.uniform(5, 8))

                if keyword_summaries:
                    combined_summaries = " ".join(keyword_summaries)
                    print("[INFO] Generating overall summary")
                    overall_summary = self.deepseek.summary_content(combined_summaries, max_tokens=4000)
                    print("[INFO] Performing overall ssentiment analysis")
                    overall_sentiment = self.perform_overall_sentiment_analysis(overall_summary)
                    aggregated_data.append([keyword_urls, overall_summary, overall_sentiment])

        finally:
            print("[INFO] Closing WebDriver")
            self.driver.quit()
            return [file_locations, aggregated_data]

def main():
    scraper = WebScraper()
    results = scraper.search_and_scrape(["start bussiness 2024 online shop snacks"])
    
    print("\nFile locations with row numbers:")
    for location in results[0]:
        print(location)
    
    print("\nAggregated Data:")
    for data in results[1]:
        print(f"\nURLs: {data[0]}")
        print(f"Overall Summary: {data[1]}")
        print(f"Summary Sentiment: {data[2]}")

if __name__ == "__main__":
    main()
#fakeuseragent==1.5.1
#selenium==4.26.1
#textblob==0.18.0
#vaderSentiment==3.3.2
