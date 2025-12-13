import threading
import time
import random
import re
from queue import Queue
from datetime import datetime
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

class EndlessCrawler(threading.Thread):
    def __init__(self, queue: Queue):
        super().__init__(daemon=True)
        self.queue = queue
        self.topics = ["science", "history", "philosophy", "technology", "mathematics", "biology", "physics","earth","ai","brains","jobs","freelance","water","alchemy","asia","europe","minecraft","roblox","anime","luffy","science","china","world","abstraction","portugal"]
        self.black_list_patterns = [
            r"porn", r"xxx", r"violence", r"hate speech",r"pussy",r"xhamster",r"nude",r"fuck"
        ]

    def is_safe(self, text: str) -> bool:
        for pattern in self.black_list_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        return True

    def run(self):
        if HAS_WIKI:
            wikipedia.set_lang("en")
            
        while True:
            try:
                text = ""
                image_urls = []
                
                if HAS_WIKI and random.random() < 0.6:
                    topic = random.choice(self.topics)
                    results = wikipedia.search(topic, results=1)
                    if results:
                        page = wikipedia.page(results[0])
                        text = page.content
                        image_urls = page.images
                
                elif HAS_DDG:
                    with DDGS() as ddgs:
                        topic = random.choice(self.topics)
                        results = [r for r in ddgs.text(topic + " explained", max_results=3)]
                        text = " ".join([r['body'] for r in results if r.get('body')])
                
                if text and len(text) > 500 and self.is_safe(text):
                    self.queue.put((text, image_urls))
                    print(f"[{datetime.now()}] Crawled {len(text)} chars — {topic}")
                
                time.sleep(random.uniform(3, 10))
                
            except Exception as e:
                time.sleep(10)
