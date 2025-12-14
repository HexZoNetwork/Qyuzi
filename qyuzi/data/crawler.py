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

class KnowledgeGapDetector:
    def find_gaps(self, topic):
        return [f"{topic} detailed mechanism", f"{topic} history", f"{topic} future implications"]

class CuriosityModel:
    def compute_scores(self, gaps):
        return [(gap, random.random()) for gap in gaps]

class CognitiveCrawler(threading.Thread):
    def __init__(self, queue: Queue):
        super().__init__(daemon=True)
        self.queue = queue
        self.topics = config.CRAWLER_TOPICS if hasattr(config, 'CRAWLER_TOPICS') else ["science", "tech"]
        self.knowledge_detector = KnowledgeGapDetector()
        self.curiosity = CuriosityModel()
        self.session = None
        try:
            import requests
            self.session = requests.Session()
            self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) QYUZI/Cognitive/1.0'})
        except ImportError:
            pass

    def clean_content(self, text):
        text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<style.*?>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        text = re.sub(r' on\w+="[^"]*"', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\S{1000,}', '', text)
        return text.strip()

    def run(self):
        print("Cognitive Crawler Started")
        while True:
            try:
                base_topic = random.choice(self.topics)
                gaps = self.knowledge_detector.find_gaps(base_topic)
                scored_gaps = self.curiosity.compute_scores(gaps)
                scored_gaps.sort(key=lambda x: x[1], reverse=True)
                
                target_query = scored_gaps[0][0]
                
                content = ""
                if HAS_DDG:
                    with DDGS() as ddgs:
                         try:
                             results = list(ddgs.text(target_query, max_results=3))
                             for r in results:
                                  raw = r.get('body', r.get('snippet', ''))
                                  content += f"\n{self.clean_content(raw)}\n"
                         except Exception:
                             pass
                
                if content and len(content) > 100:
                    with queue_lock:
                         self.queue.put((content, []))
                    print(f"Cognitive Crawl: Ingested {len(content)} chars for '{target_query}'")
                else:
                    pass

            except Exception as e:
                print(f"Cognitive Crawler Error: {e}")
            
            time.sleep(2)
