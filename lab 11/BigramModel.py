import re
import math
from collections import Counter

class BigramModel:
    def __init__(self, text=None, model_filepath=None):
        self.model = {}
        if text:
            self.train(text)
        elif model_filepath:
            self.load(model_filepath)

    def train(self, text):
        self.model = self.get_bigrams(text)

    def get_bigrams(self, text):
        cleaned_text = re.sub(r'[^a-zA-Z]', '', text.lower())
        bigrams = [cleaned_text[i:i + 2] for i in range(len(cleaned_text) - 1)]
        return Counter(bigrams)

    def save(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for bigram, count in self.model.items():
                f.write(f"{bigram}: {count}\n")

    def load(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                bigram, count = line.strip().split(': ')
                self.model[bigram] = int(count)

    def calculate_likelihood(self, text):
        bigrams = self.get_bigrams(text)
        log_likelihood = 0
        for bigram, count in bigrams.items():
            freq = self.model.get(bigram, 0)
            log_likelihood += count * math.log(freq if freq > 0 else 1e-10)

        num_bigrams = sum(bigrams.values())
        return log_likelihood / num_bigrams if num_bigrams > 1 else log_likelihood