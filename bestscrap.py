import time
import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote, unquote, parse_qs, urlparse
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys  # Added for sending keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deepseek import DeepSeek
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import concurrent.futures
import threading

class Logger:
    def __init__(self):
        self.start_times = {}
        self.pending_messages = {}
        self.total_start_time = None
        self.current_progress = 0
        self.total_progress = 0
        self.lock = threading.Lock()
        
    def set_total_progress(self, keywords_count, results_per_keyword):
        with self.lock:
            self.total_progress = keywords_count * (results_per_keyword + 1)
            print(f"[PROGRESS] {self.current_progress}/{self.total_progress}")
        
    def update_progress(self):
        with self.lock:
            self.current_progress += 1
            print(f"[PROGRESS] {self.current_progress}/{self.total_progress}")
        
    def start_total_timer(self):
        with self.lock:
            self.total_start_time = time.time()
        self.log("Starting total execution timer")
        
    def end_total_timer(self):
        with self.lock:
            if self.total_start_time:
                total_elapsed = time.time() - self.total_start_time
        self.log(f"Total execution time: {total_elapsed:.2f} seconds")
        
    def start(self, message):
        with self.lock:
            self.start_times[message] = time.time()
            self.pending_messages[message] = message
        print(f"[INFO] {message}", end='', flush=True)
        
    def end(self, message, word_count=None):
        with self.lock:
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
            },
            {
                'name': 'Qwant',
                'url': 'https://www.qwant.com/?q=',
                'result_selector': '.result__url',
                'title_selector': '.result__title',
                'snippet_selector': '.result__desc'
            },
            {
                'name': 'DuckDuckGo',
                'url': 'https://html.duckduckgo.com/html/?q=',
                'result_selector': 'a.result__a',
                'title_selector': 'a.result__a',
                'snippet_selector': 'a.result__snippet'
            }
        ]
        self.ua = UserAgent()
        self.setup_selenium()
        self.setup_sentiment_analyzers()
        self.engine_failures = 0
        self.max_engine_failures = len(self.search_engines)
        self.scraped_urls = set()
        self.warning = []
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.futures = []

        self.deepseek_enabled = False  # Will be set based on summary options

        try:
            self.deepseek = DeepSeek()
        except Exception as e:
            self.logger.warning(f"Failed to initialize DeepSeek: {str(e)}")
            self.warning.append(f"Failed to initialize DeepSeek: {str(e)}")
            self.deepseek = None

    def extract_duckduckgo_url(self, href):
        """Extract actual URL from DuckDuckGo redirect link"""
        try:
            if "duckduckgo.com/l/?" in href:
                parsed = urlparse(href)
                params = parse_qs(parsed.query)
                if 'uddg' in params:
                    return unquote(params['uddg'][0])
            return href
        except Exception as e:
            self.logger.error(f"Failed to extract DuckDuckGo URL: {str(e)}")
            return href

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
            self.warning.append(f"Summary generation failed: {str(e)}")
            return content

    def setup_sentiment_analyzers(self):
        self.logger.start("Setting up sentiment analyzers")
        try:
            nltk.download('punkt', quiet=True)
        except:
            self.logger.warning("NLTK punkt download failed, but continuing...")
            self.warning.append("NLTK punkt download failed, but continuing...")
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
        # Adjust headless option depending on your need
        chrome_options.add_argument('--headless=new')
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

        # Set window size for headless mode
        chrome_options.add_argument('--window-size=1920,1080')

        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": self.ua.random})
        self.logger.end("Setting up Selenium WebDriver")

    def fetch_with_selenium(self, url, retry_count=2):
        for attempt in range(retry_count):
            try:
                self.driver.set_page_load_timeout(30)
                self.driver.get(url)
                try:
                    WebDriverWait(self.driver, 15).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )
                except Exception as wait_error:
                    self.logger.error(f"Page load timeout: {str(wait_error)}")
                    continue
                time.sleep(random.uniform(3, 5))
                if "Error" in self.driver.title or "404" in self.driver.title:
                    raise Exception("Error page detected")
                page_text = self.driver.find_element(By.TAG_NAME, "body").text
                if not page_text.strip():
                    raise Exception("Empty page content")
                return self.driver.page_source
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(random.uniform(2, 4))
                    continue
        return None
    
    def scrape_website_content(self, url):
        """Method using LangChain for web scraping with word count validation"""
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter()
            documents = text_splitter.split_documents(docs)
            content = "\n\n".join(doc.page_content for doc in documents)
            word_count = len(content.split())
            if word_count < 500:
                self.logger.end(f"Fetching content from: {url}", word_count)
                self.logger.warning(f"Content from {url} has insufficient words ({word_count} words)")
                self.warning.append(f"Content from {url} has insufficient words ({word_count} words)")
                return None
            return content
        except Exception as e:
            self.logger.error(f"LangChain scraping failed: {str(e)}")
            try:
                content = self.fetch_with_selenium(url)
                if content:
                    word_count = len(content.split())
                    if word_count < 500:
                        self.logger.end(f"Fetching content from: {url}", word_count)
                        self.logger.warning(f"Selenium content from {url} has insufficient words ({word_count} words)")
                        self.warning.append(f"Selenium content from {url} has insufficient words ({word_count} words)")
                        return None
                    return content
            except Exception as selenium_error:
                self.logger.error(f"Selenium scraping failed: {str(selenium_error)}")
                return None
        
    def analyze_results(self, keyword_summaries, keywords, summary):
        if not keyword_summaries:
            return "", ""
        overall_summary = "\n\n\n".join(keyword_summaries)
        if summary[1] and self.deepseek_enabled:
            combined_summaries = " ".join(keyword_summaries)
            combined_word_count = len(combined_summaries.split())
            self.logger.start("Generating overall summary")
            future_overall_summary = self.executor.submit(self.deepseek.summary_content, combined_summaries, keywords, max_tokens=4000)
            overall_summary = future_overall_summary.result()
            summary_word_count = len(overall_summary.split())
            self.logger.end("Generating overall summary", f"{combined_word_count}->{summary_word_count}")
        self.logger.start("Performing overall sentiment analysis")
        overall_sentiment = self.perform_overall_sentiment_analysis(overall_summary)
        self.logger.end("Performing overall sentiment analysis")
        return overall_summary, overall_sentiment

    def scrape_google_search(self, keyword, num_results=3):
        self.logger.log(f"Using Google search for keyword: {keyword}")
        scraped_urls = set()
        duplicate_count = {}  # Track duplicate domains
        keyword_urls = []
        keyword_summaries = []
        keyword_contents = []
        results = []

        try:
            self.logger.start("Fetching Google search page")
            self.driver.get('https://www.google.com')
            time.sleep(5)
            self.logger.end("Fetching Google search page")
            
            # Handle cookie consent
            try:
                consent_button = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Accept all')]"))
                )
                consent_button.click()
                time.sleep(2)
            except TimeoutException:
                pass
            
            # Find and use search box
            self.logger.start("Finding search box")
            search_box = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'textarea[name="q"], input[name="q"]'))
            )
            self.logger.end("Finding search box")
            
            search_box.clear()
            search_box.send_keys(keyword)
            search_box.send_keys(Keys.RETURN)
            self.logger.log(f"Searching for: {keyword}")
            time.sleep(5)
            
            # Get search results
            search_results = []
            selectors = ["div.MjjYud", "div.g", "div[data-sokoban-container]"]
            
            for selector in selectors:
                try:
                    self.logger.start(f"Finding search results with selector: {selector}")
                    search_results = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
                    )
                    self.logger.end(f"Finding search results with selector: {selector}")
                    if search_results:
                        break
                except:
                    self.logger.warning(f"No search results found with selector: {selector}")
                    continue
            
            if not search_results:
                self.logger.warning("No search results found with any selector")
                return [], [], []
            
            # Process results until we have desired number of unique ones
            result_index = 0
            while len(keyword_urls) < num_results and result_index < len(search_results):
                try:
                    result = search_results[result_index]
                    result_index += 1
                    
                    # Find link
                    link_element = None
                    for link_selector in ["a[href]", "a[data-ved]", "a[ping]"]:
                        try:
                            link_element = result.find_element(By.CSS_SELECTOR, link_selector)
                            break
                        except:
                            continue
                    
                    if not link_element:
                        continue
                        
                    link = link_element.get_attribute("href")
                    
                    # Skip invalid or duplicate URLs
                    if not link or link.startswith(('javascript:', 'about:')):
                        continue
                        
                    # Extract base domain for duplicate checking
                    base_url = urlparse(link).netloc
                    
                    # Check if we've seen this domain too many times
                    if base_url in duplicate_count:
                        duplicate_count[base_url] += 1
                        if duplicate_count[base_url] >= 3:
                            self.logger.log(f"Skipping {base_url} - seen too many times")
                            continue
                    else:
                        duplicate_count[base_url] = 1
                        
                    # Skip if we've already scraped this exact URL
                    if link in self.scraped_urls:
                        self.logger.log(f"Skipping duplicate URL: {link}")
                        continue
                        
                    self.logger.start(f"Fetching content from: {link}")
                    content = self.scrape_website_content(link)
                    if content:
                        word_count = len(content.split())
                        self.logger.end(f"Fetching content from: {link}", word_count)
                        
                        self.scraped_urls.add(link)
                        if self.deepseek_enabled:
                            future_summary = self.executor.submit(self.get_content_summary, content[:10000], [keyword], link)
                            keyword_summaries.append(future_summary)
                        else:
                            keyword_summaries.append(content)
                        keyword_urls.append(link)
                        keyword_contents.append(content)
                        self.logger.update_progress()
                    else:
                        self.logger.end(f"Fetching content from: {link}")
                        self.logger.warning(f"Skipping {link} due to insufficient content")
                    time.sleep(2)
                    # No need to navigate back
                except Exception as e:
                    self.logger.error(f"Could not process result: {str(e)}")
                    continue
            # Collect summaries if any
            for i, item in enumerate(keyword_summaries):
                if isinstance(item, concurrent.futures.Future):
                    keyword_summaries[i] = item.result()
            return keyword_urls, keyword_summaries, keyword_contents

        except Exception as e:
            self.logger.error(f"Error during Google search scraping: {str(e)}")
            return [], [], []

    def search_and_scrape(self, keywords=["business strategy 2024 startup profit"], num_res=3, num_attempt=1, summary=[True,False]):
        results = []
        keywords = [" ".join(keywords)]
        self.logger.set_total_progress(len(keywords), num_res)
        self.deepseek_enabled = summary[0]  # Enable or disable deepseek based on summary option
        try:
            for keyword in keywords:
                self.logger.log(f"Processing keyword: {keyword}")
                # First, try to get URLs via Google search
                keyword_urls, keyword_summaries, keyword_contents = self.scrape_google_search(keyword, num_results=num_res)
                if len(keyword_urls) >= num_res:
                    self.logger.log(f"Found sufficient URLs via Google search for keyword: {keyword}")
                    # Proceed to analyze results
                    overall_summary, overall_sentiment = self.analyze_results(keyword_summaries, [keyword], summary)
                    results.append({
                        'keyword': keyword,
                        'urls': keyword_urls,
                        'content': keyword_contents,
                        'overall_summary': overall_summary,
                        'sentiment': overall_sentiment,
                        'warning': self.warning
                    })
                else:
                    self.logger.log(f"Not enough results from Google search, trying multi-engine search")
                    # Reset the summaries and contents
                    keyword_summaries = []
                    keyword_contents = []
                    content_summary_futures = []
                    results_count = 0
                    self.engine_failures = 0
                    engine_index = 0
                    while results_count < num_res and engine_index < len(self.search_engines):
                        engine = self.search_engines[engine_index]
                        
                        success = False
                        for attempt in range(num_attempt):
                            self.logger.log(f"Trying search engine: {engine['name']} (Attempt {attempt + 1}/{num_attempt})")
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
                                    
                                    if engine['name'] == 'DuckDuckGo':
                                        href = self.extract_duckduckgo_url(href)

                                    if href in self.scraped_urls or not href or not href.startswith('http'):
                                        continue

                                    try:
                                        self.logger.start(f"Fetching content from: {href}")
                                        content = self.scrape_website_content(href)
                                        
                                        if not content:
                                            self.logger.warning(f"Skipping {href} due to insufficient content")
                                            self.warning.append(f"Skipping {href} due to insufficient content")
                                            continue
                                            
                                        word_count = len(content.split())
                                        self.logger.end(f"Fetching content from: {href}", word_count)
                                        
                                        self.scraped_urls.add(href)
                                        if summary[0] and self.deepseek_enabled:
                                            future_summary = self.executor.submit(self.get_content_summary, content[:10000], [keyword], href)
                                            content_summary_futures.append(future_summary)
                                            keyword_summaries.append(future_summary)
                                        else:
                                            keyword_summaries.append(content)

                                        keyword_urls.append(href)
                                        keyword_contents.append(content)
                                        results_count += 1
                                        success = True
                                        self.logger.update_progress()
                                            
                                    except Exception as e:
                                        self.logger.error(f"Failed to fetch page content: {str(e)}")
                                    
                                    time.sleep(random.uniform(2, 4))

                                if success:
                                    break

                            except Exception as e:
                                self.logger.error(f"{engine['name']} search failed: {str(e)}")
                            
                            if attempt < num_attempt - 1:
                                time.sleep(random.uniform(3, 5))

                        if not success:
                            self.engine_failures += 1
                            
                        engine_index += 1
                        
                    if results_count < num_res:
                        self.logger.warning(f"Failed to find more potential results on possible engines")
                        self.warning.append(f"Failed to find more potential results on possible engines")

                    # Collect the results from the content summary futures
                    for i, item in enumerate(keyword_summaries):
                        if isinstance(item, concurrent.futures.Future):
                            keyword_summaries[i] = item.result()

                    overall_summary, overall_sentiment = self.analyze_results(keyword_summaries, [keyword], summary)
                    results.append({
                        'keyword': keyword,
                        'urls': keyword_urls,
                        'content': keyword_contents,
                        'overall_summary': overall_summary,
                        'sentiment': overall_sentiment,
                        'warning': self.warning
                    }) 
        finally:
            self.driver.quit()
            self.executor.shutdown(wait=True)
        return results

def scrapfast(words=["business strategy 2024 startup profit"], num_res=3, num_attempt=1, summary=[True, False]):
    try:
        scraper = WebScraper()
        scraper.logger.start_total_timer()
        
        results = scraper.search_and_scrape(words, num_res, num_attempt, summary)
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
    results = scrapfast(["online", "snack shop 2024", "business idea strategy", "profit start up"])
    print("\nResults:")
    for result in results:
        print(f"\nKeyword: {result['keyword']}")
        print(f"URLs: {result['urls']}")
        # print(f"Contents: {result['content']}")
        print(f"Overall Summary: {result['overall_summary'][:10000]}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Warning: {result['warning']}")