import re
import random
from collections import Counter

class SentenceGenerator:
    def __init__(self, text):
        self.alphabet_probs = self.build_alphabet_probs(text)

    def build_alphabet_probs(self, text):
        cleaned_text = re.sub(r'[^a-zA-Z]', '', text.lower())
        total_chars = len(cleaned_text)
        char_counts = Counter(cleaned_text)
        return {char: count / total_chars for char, count in char_counts.items()}

    def generate_random_sentence(self, length_range=(50, 250)):
        population, weights = zip(*self.alphabet_probs.items())
        length = random.randint(*length_range)
        return ''.join(random.choices(population, weights=weights, k=length))