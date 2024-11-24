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
        
    def end(self, message, word_count=None):
        if message in self.start_times:
            elapsed = time.time() - self.start_times[message]
            if message in self.pending_messages:
                output = f" ({elapsed:.2f}s)"
                if word_count:
                    if isinstance(word_count, str) and "->" in word_count:
                        output = f" (Summarized {word_count} words in {elapsed:.2f}s)"
                    else:
                        output = f" ({word_count} words in {elapsed:.2f}s)"
                print(output)
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
        self.engine_failures = 0
        self.max_engine_failures = len(self.search_engines)

        try:
            self.deepseek = DeepSeek()
        except Exception as e:
            self.logger.warning(f"Failed to initialize DeepSeek: {str(e)}")
            self.deepseek = None

    def get_content_summary(self, content, keywords, url):
        if self.deepseek is None:
            return content
                
        try:
            word_count = len(content.split())
            self.logger.start(f"Generating content summary for keywords: {keywords} in {url}")
            summary = self.deepseek.summary_web(content, keywords, max_tokens=4000)
            summary_word_count = len(summary.split())
            self.logger.end(f"Generating content summary for keywords: {keywords} in {url}", 
                        f"{word_count}->{summary_word_count}")
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
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-setuid-sandbox')
        chrome_options.add_argument('--disable-logging')
        chrome_options.add_argument('--log-level=3')
        chrome_options.add_argument('--disable-images')
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": self.ua.random})
        self.logger.end("Setting up Selenium WebDriver")

    # Add better error handling in the fetch_with_selenium method:
    def fetch_with_selenium(self, url, retry_count=2):
        for attempt in range(retry_count):
            try:
                msg = f"Fetching URL (attempt {attempt + 1}): {url}"
                self.logger.start(msg)
                
                # Add timeout parameter
                self.driver.set_page_load_timeout(30)
                self.driver.get(url)
                
                # Add more specific wait conditions
                try:
                    WebDriverWait(self.driver, 15).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )
                except Exception as wait_error:
                    self.logger.error(f"Page load timeout: {str(wait_error)}")
                    continue
                    
                time.sleep(random.uniform(3, 5))
                
                # Verify page loaded successfully
                if "Error" in self.driver.title or "404" in self.driver.title:
                    raise Exception("Error page detected")
                    
                page_text = self.driver.find_element(By.TAG_NAME, "body").text
                if not page_text.strip():
                    raise Exception("Empty page content")
                    
                word_count = len(page_text.split())
                self.logger.end(msg, word_count)
                return self.driver.page_source
                
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(random.uniform(2, 4))
                    continue
        return None
    
    def analyze_results(self, keyword_summaries, keywords):
        if not keyword_summaries:
            return "", ""
            
        combined_summaries = " ".join(keyword_summaries)
        combined_word_count = len(combined_summaries.split())
        self.logger.start("Generating overall summary")
        overall_summary = self.deepseek.summary_content(combined_summaries, keywords, max_tokens=4000)
        summary_word_count = len(overall_summary.split())
        self.logger.end("Generating overall summary", f"{combined_word_count}->{summary_word_count}")
        self.logger.start("Performing overall sentiment analysis")
        overall_sentiment = self.perform_overall_sentiment_analysis(overall_summary)
        self.logger.end("Performing overall sentiment analysis")
        return overall_summary, overall_sentiment

    def search_and_scrape(self, keywords=["business strategy 2024 startup profit"], num_res=3, num_attempt=3):
        results = []
        keywords = [" ".join(keywords)]
        self.logger.set_total_progress(len(keywords), num_res)
        
        try:
            for keyword in keywords:
                self.logger.log(f"Processing keyword: {keyword}")
                results_count = 0
                keyword_urls = []
                keyword_summaries = []
                self.engine_failures = 0

                # Continue trying engines until we get enough results or all engines fail
                engine_index = 0
                while results_count < num_res and engine_index < len(self.search_engines):
                    engine = self.search_engines[engine_index]
                    
                    success = False
                    for attempt in range(num_attempt):
                        self.logger.log(f"Trying search engine: {engine['name']} (Attempt {attempt + 1}/3)")
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
                                            
                                            content_summary = self.get_content_summary(content[:10000], keywords, href)
                                            
                                            keyword_urls.append(href)
                                            keyword_summaries.append(content_summary)
                                            
                                            results_count += 1
                                            success = True
                                            self.logger.update_progress()
                                            
                                    except Exception as e:
                                        self.logger.error(f"Failed to fetch page content: {str(e)}")
                                
                                time.sleep(random.uniform(4, 7))

                            if success:
                                break  # Exit retry loop if results were found
                                
                        except Exception as e:
                            self.logger.error(f"{engine['name']} search failed: {str(e)}")
                        
                        if attempt < num_attempt - 1:
                            time.sleep(random.uniform(5, 8))

                    if not success:
                        self.engine_failures += 1
                        
                    engine_index += 1

                # Process results for this keyword if any were found
                overall_summary, overall_sentiment = self.analyze_results(keyword_summaries, keywords)

                results.append({
                    'keyword': keyword,
                    'urls': keyword_urls,
                    'overall_summary': overall_summary,
                    'sentiment': overall_sentiment
                })

        finally:
            self.logger.log("Closing WebDriver")
            self.driver.quit()
            
        if not results:
            self.logger.warning("No results found for any keywords")
        return results

def scrapfast(words=["business strategy 2024 startup profit"], num_res=3, num_attempt=3):
    try:
        scraper = WebScraper()
        scraper.logger.start_total_timer()
        
        results = scraper.search_and_scrape(words, num_res, num_attempt)
        scraper.logger.end_total_timer()
        
        if not results:
            scraper.logger.warning("Search completed but no results were found")
            return ""
        return results
            
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
    results = scrapfast(["online", "snack shop 2024", "business idea strategy", " profit start up"],3,3)
    print("\nResults:")
    for result in results:
        print(f"\nKeyword: {result['keyword']}")
        print(f"URLs: {result['urls']}")
        print(f"Overall Summary: {result['overall_summary']}")
        print(f"Sentiment: {result['sentiment']}")
