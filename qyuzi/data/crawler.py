import threading
import time
import random
import re
from queue import Queue
from datetime import datetime
from qyuzi.config import config
queue_lock = threading.Lock()
try:
    import wikipedia
    HAS_WIKI = True
except ImportError:
    HAS_WIKI = False

try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False

class WebCrawler:
    def __init__(self):
        self.session = None
        try:
            import requests
            self.session = requests.Session()
            self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) QYUZI/1.0'})
        except ImportError:
            print("Requests library not found. Web crawling disabled.")

    def fetch(self, url):
        if not self.session: return ""
        try:
            resp = self.session.get(url, timeout=5)
            if resp.status_code == 200:
                return resp.text
            return ""
        except Exception as e:
            print(f"Fetch Error {url}: {e}")
            return ""

    def search_and_scrape(self, topic):
        content = ""
        images = []
        
        if HAS_WIKI:
            try:
                results = wikipedia.search(topic, results=1)
                if results:
                    page = wikipedia.page(results[0], auto_suggest=False)
                    content += f"\n--- {page.title} ---\n{page.content}\n"
                    images.extend(page.images)
            except Exception as e:
                print(f"Wiki Error ({topic}): {e}")

        if HAS_DDG:
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(f"{topic} detailed explanation", max_results=2))
                    for r in results:
                        content += f"\n{r['body']}\n"
            except Exception as e:
                print(f"DDG Error ({topic}): {e}")
                
        return content, images

class EndlessCrawler(threading.Thread):
    def __init__(self, queue: Queue):
        super().__init__(daemon=True)
        self.queue = queue
        self.topics = ["science", "history", "philosophy", "technology", "mathematics", "biology", "physics", "ai", "quantum mechanics", "neuroscience", "space exploration"]
        self.black_list = [r"porn", r"xxx", r"violence", r"hate speech"]
        self.web = WebCrawler()
        self.loader = LocalFileLoader()

    def is_safe(self, text):
        for p in self.black_list:
            if re.search(p, text, re.IGNORECASE):
                return False
        return True

    def run(self):
        print("Crawler Service-Background Started")
        while True:
            text_batch = ""
            img_batch = []
            
            try:
                topic = random.choice(self.topics)
                web_text, web_imgs = self.web.search_and_scrape(topic)
                if web_text:
                    text_batch += web_text
                    img_batch.extend(web_imgs)
            except Exception as e:
                print(f"Crawler Main Loop Error: {e}")

            local = self.loader.get_chunk()
            if local:
                text_batch += "\n" + local

            if len(text_batch) > 500 and self.is_safe(text_batch):
                self.queue.put((text_batch, img_batch))
                print(f"Ingested {len(text_batch)} chars for {topic}")
            else:
                print(f"Yield low for {topic}, retrying...")
            
            time.sleep(2)
