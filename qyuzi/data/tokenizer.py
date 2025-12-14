try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

class SimpleTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.eot_token = 256 

    def encode(self, text, allowed_special=None):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")

class AutoTokenizer:
    _instance = None
    
    @staticmethod
    def get_instance():
        if AutoTokenizer._instance is None:
            if HAS_TIKTOKEN:
                try:
                    AutoTokenizer._instance = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    AutoTokenizer._instance = SimpleTokenizer()
            else:
                AutoTokenizer._instance = SimpleTokenizer()
        return AutoTokenizer._instance

def encode(text):
    tokenizer = AutoTokenizer.get_instance()
    if isinstance(tokenizer, SimpleTokenizer):
        return tokenizer.encode(text)
    return tokenizer.encode(text, allowed_special={'<|endoftext|>'})

def decode(ids):
    tokenizer = AutoTokenizer.get_instance()
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(ids)
    return ""
