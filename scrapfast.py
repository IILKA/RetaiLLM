import time
import random
import requests
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
from datetime import datetime

class Logger:
    def __init__(self):
        self.start_times = {}
        self.pending_messages = {}
        self.total_start_time = None
        self.current_progress = 0
        self.total_progress = 0
        
    def set_total_progress(self, keywords_count, results_per_keyword):
        # Each keyword gets results_per_keyword results + 1 overall summary
        self.total_progress = keywords_count * (results_per_keyword + 1)
        print(f"[PROGRESS] {self.current_progress}/{self.total_progress}")
        
    def update_progress(self):
        self.current_progress += 1 
        print(f"[PROGRESS] {self.current_progress}/{self.total_progress}")
        
    def start_total_timer(self):
        self.total_start_time = time.time()
        self.log("Starting total execution timer")
        
    def end_total_timer(self):
        if self.total_start_time:
            total_elapsed = time.time() - self.total_start_time
            self.log(f"Total execution time: {total_elapsed:.2f} seconds")
        
    def start(self, message):
        self.start_times[message] = time.time()
        self.pending_messages[message] = message
        print(f"[INFO] {message}", end='', flush=True)
        
    def end(self, message):
        if message in self.start_times:
            elapsed = time.time() - self.start_times[message] 
            if message in self.pending_messages:
                if "Fetching content from:" in message:
                    print("")
                else:
                    print(f" ({elapsed:.2f}s)")
                del self.pending_messages[message]
            del self.start_times[message]
            
    def log(self, message):
        print(f"[INFO] {message}")
            
    def warning(self, message):
        print(f"[WARNING] {message}")
        
    def error(self, message):
        print(f"[ERROR] {message}")

class WebScraper:
    def __init__(self):
        self.logger = Logger()
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
        self.setup_selenium()
        self.setup_sentiment_analyzers()

        try:
            self.deepseek = DeepSeek()
        except Exception as e:
            self.logger.warning(f"Failed to initialize DeepSeek: {str(e)}")
            self.deepseek = None

    def get_content_summary(self, content):
        if self.deepseek is None:
            return content
            
        try:
            summary = self.deepseek.summary_web(content, max_tokens=4000)
            return summary
        except Exception as e:
            self.logger.warning(f"Summary generation failed: {str(e)}")
            return content

    def setup_sentiment_analyzers(self):
        self.logger.start("Setting up sentiment analyzers")
        try:
            nltk.download('punkt', quiet=True)
        except:
            self.logger.warning("NLTK punkt download failed, but continuing...")
        self.vader = SentimentIntensityAnalyzer()
        self.logger.end("Setting up sentiment analyzers")

    def perform_overall_sentiment_analysis(self, text):
        try:
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            vader_scores = self.vader.polarity_scores(text)
            
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
            
            return sentiment_summary
                
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {str(e)}")
            return {'error': str(e)}

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
        self.logger.start("Setting up Selenium WebDriver")
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
        self.logger.end("Setting up Selenium WebDriver")

    def fetch_with_selenium(self, url, retry_count=2):
        for attempt in range(retry_count):
            try:
                msg = f"Fetching URL (attempt {attempt + 1}): {url}"
                self.logger.start(msg)
                self.driver.get(url)
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                time.sleep(random.uniform(3, 5))
                self.logger.end(msg)
                return self.driver.page_source
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(random.uniform(2, 4))
                    continue
        return None

    def search_and_scrape(self, keywords=["business strategy 2024 startup profit"], num_res=3):
        results = []
        self.logger.set_total_progress(len(keywords), num_res)
        try:
            for keyword in keywords:
                self.logger.log(f"Processing keyword: {keyword}")
                results_count = 0
                keyword_urls = []
                keyword_summaries = []

                for engine in self.search_engines:
                    if results_count >= num_res:
                        break

                    self.logger.log(f"Trying search engine: {engine['name']}")
                    try:
                        search_url = f"{engine['url']}{quote(keyword)}"
                        page_source = self.fetch_with_selenium(search_url)
                        
                        if not page_source:
                            continue

                        soup = BeautifulSoup(page_source, 'html.parser')
                        links = soup.select(engine['result_selector'])
                        
                        if not links:
                            continue

                        self.logger.log(f"Found {len(links)} potential results on {engine['name']}")
                        
                        for link in links:
                            if results_count >= num_res:
                                break
                                
                            href = link.get('href')
                            if href and href.startswith('http'):
                                try:
                                    self.logger.log(f"Fetching content from: {href}")
                                    page_content = self.fetch_with_selenium(href)
                                    
                                    if page_content:
                                        page_soup = BeautifulSoup(page_content, 'html.parser')
                                        content = ' '.join(page_soup.get_text(strip=True).split())
                                        
                                        content_summary = self.get_content_summary(content)
                                        
                                        keyword_urls.append(href)
                                        keyword_summaries.append(content_summary)
                                        
                                        results_count += 1
                                        self.logger.update_progress()
                                        self.logger.end(f"Fetching content from: {href}")
                                    
                                except Exception as e:
                                    self.logger.error(f"Failed to fetch page content: {str(e)}")
                            
                            time.sleep(random.uniform(4, 7))
                    
                    except Exception as e:
                        self.logger.error(f"{engine['name']} search failed: {str(e)}")
                    
                    time.sleep(random.uniform(5, 8))

                if keyword_summaries:
                    combined_summaries = " ".join(keyword_summaries)
                    self.logger.start("Generating overall summary")
                    overall_summary = self.deepseek.summary_content(combined_summaries, max_tokens=4000)
                    self.logger.end("Generating overall summary")
                    
                    self.logger.start("Performing overall sentiment analysis")
                    overall_sentiment = self.perform_overall_sentiment_analysis(overall_summary)
                    self.logger.end("Performing overall sentiment analysis")
                    
                    self.logger.update_progress()
                    
                    results.append({
                        'keyword': keyword,
                        'urls': keyword_urls,
                        'overall_summary': overall_summary,
                        'sentiment': overall_sentiment
                    })

        finally:
            self.logger.log("Closing WebDriver")
            self.driver.quit()
            return results if results else ""

def scrapfast(words=["business strategy 2024 startup profit"], num_res=3):
    try:
        scraper = WebScraper()
        scraper.logger.start_total_timer()
        
        results = scraper.search_and_scrape(words, num_res)
        scraper.logger.end_total_timer()
        return results if results else ""
            
    except ImportError as e:
        print(f"\nWebScraper module not found: {str(e)}")
        return ""
        
    except ConnectionError as e:
        print(f"\nNetwork connection error: {str(e)}")
        return ""
        
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        return ""

if __name__ == "__main__":
    results = scrapfast(["online snack shop 2024 business idea strategy profit start up", "hong kong 2024 startup company new clothing fasion reminder notes"],3)
    print("\nResults:")
    for result in results:
        print(f"\nKeyword: {result['keyword']}")
        print(f"URLs: {result['urls']}")
        print(f"Overall Summary: {result['overall_summary']}")
        print(f"Sentiment: {result['sentiment']}")
