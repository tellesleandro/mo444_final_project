from features_generator_bow import *
from collections import Counter,defaultdict

import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', \
                    level=logging.INFO, \
                    datefmt='%Y-%m-%d %H:%M:%S')

from pdb import set_trace as bp

class Statistics:

    def __init__(self, X, y):
        logging.info('statistics.init')
        self.__X = X
        self.__y = y
        self.__fg = FeaturesGenerator()
        self.cleaned_corpus, self.cleaned_labels = self.__fg.clean_corpus(X, y)
        self.features, self.labels = self.__fg.fit(X, y)
        self.vocabulary = self.__fg.vocabulary()
        self.reversed_vocabulary = dict((v, k) for k, v in self.vocabulary.items())

    def generate_corpus_with_score(self):
        logging.info('statistics.generate_corpus_with_score')
        self.corpus_with_score = []
        for row_idx, row in enumerate(self.features):
            word_scores = self.indices_score(row.indices, row.data)
            self.corpus_with_score.append(word_scores)

    def generate_labels_with_score(self):
        logging.info('statistics.generate_labels_with_score')
        self.labels_with_score = {}
        for row_idx, row in enumerate(self.features):
            label = self.labels[row_idx]
            if label not in self.labels_with_score:
                self.labels_with_score[label] = {}
            word_scores = self.indices_score(row.indices, row.data)
            for word, score in word_scores.items():
                if word not in self.labels_with_score[label] or \
                            self.labels_with_score[label][word] < score:
                    self.labels_with_score[label][word] = score

    def indices_score(self, indices, data):
        word_scores = {}
        for indice_idx, indice in enumerate(indices):
            word = self.reversed_vocabulary[indice]
            score = data[indice_idx]
            word_scores[word] = score
        return word_scores

    def generate_labels_with_frequencies(self):
        logging.info('statistics.generate_labels_with_frequencies')
        self.labels_with_frequencies = {}
        for idx, document in enumerate(self.cleaned_corpus):
            label = self.cleaned_labels[idx]
            if label not in self.labels_with_frequencies:
                self.labels_with_frequencies[label] = {}
            word_frequencies = Counter(document.split())
            for word, frequency in word_frequencies.items():
                if word not in self.labels_with_frequencies[label] or \
                            self.labels_with_frequencies[label][word] < frequency:
                    self.labels_with_frequencies[label][word] = frequency
