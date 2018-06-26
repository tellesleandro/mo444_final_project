import os
import re
import html
import nltk
import numpy as np
from math import log
from collections import Counter,defaultdict
from scipy import sparse
import pickle

import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', \
                    level=logging.INFO, \
                    datefmt='%Y-%m-%d %H:%M:%S')

from pdb import set_trace as bp

class FeaturesGeneratorBoW:

    STRIP_SEQUENCE = "([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})|[!#$%^&*()\_\-–+={}\[\]\\\:;'<,>\.?/`´“”²³\|" + '"]'

    def build_model(self, corpus, labels):
        logging.info('features_generator_bow.build_model')
        self.corpus, self.labels = self.clean_corpus(corpus, labels)

        self.words = []
        self.classes = []
        self.bow = []

        for idx, document in enumerate(self.corpus):
            label = self.labels[idx]
            words = self.words_from_text(document)
            self.words.extend(words)
            if label not in self.classes:
                self.classes.append(label)
            self.bow.append((words, label))

        self.words = list(set(self.words))

        self.bow_model = [0] * len(self.words)
        self.labels_model = [0] * len(self.classes)

    def generate_features_and_targets_for_model(self):
        logging.info('features_generator_bow.generate_features_and_targets_for_model')
        self.model_features = []
        self.model_targets = []
        for words, label in self.bow:
            bow_from_words = self.bow_from_words(words)
            target = self.target_from_label(label)
            self.model_features.append(bow_from_words)
            self.model_targets.append(target)
        self.model_features = np.array(self.model_features)
        self.model_targets = np.array(self.model_targets)

    def words_from_text(self, text):
        logging.info('features_generator_bow.words_from_text')
        text = self.clean_text(text)
        words = nltk.word_tokenize(text)
        n_grams = self.n_grams(words, 2)
        words.extend(n_grams)
        print(words)
        return words

    def bow_from_text(self, text):
        logging.info('features_generator_bow.bow_from_text')
        words = self.words_from_text(text)
        return self.bow_from_words(words)

    def bow_from_words(self, words):
        logging.info('features_generator_bow.bow_from_words')
        bow = []
        for word in self.words:
            bow.append(1) if word in words else bow.append(0)
        return np.array(bow)

    def target_from_label(self, label):
        logging.info('features_generator_bow.target_from_label')
        one_hot_encoded_label = list(self.labels_model)
        one_hot_encoded_label[self.classes.index(label)] = 1
        return np.array(one_hot_encoded_label)

    def clean_corpus(self, corpus, labels):
        logging.info('features_generator_bow.clean_corpus')
        new_corpus = []
        new_labels = []
        for idx, document in corpus.items():
            document = self.clean_text(document)
            if len(document) == 0: continue
            new_corpus.append(document)
            new_labels.append(labels[idx])
        return new_corpus, new_labels

    def clean_text(self, text):
        logging.info('features_generator_bow.clean_text')
        if not isinstance(text, str) or text is None or len(text) == 0:
            return ''
        previous_text = text
        while True:
            actual_text = html.unescape(previous_text)
            if actual_text == previous_text: break
            previous_text = actual_text
        while True:
            actual_text = self.remove_tags(previous_text)
            if actual_text == previous_text: break
            previous_text = actual_text
        actual_text = re.sub(self.STRIP_SEQUENCE, " ", previous_text.lower()).strip()
        valid_words = []
        for word in actual_text.split():
            if len(word) > 3 and not word.isdigit():
                valid_words.append(word)
        actual_text = ' '.join(valid_words)
        return actual_text

    def remove_tags(self, text):
        logging.info('features_generator_bow.remove_tags')
        text = re.sub('<.*?>', '', text)
        text = re.sub('\s+', ' ', text ).strip()
        return text

    def n_grams(self, input_list, n):
        n_grams = []
        n_grams_zipped = zip(*[input_list[i:] for i in range(n)])
        for words in n_grams_zipped:
            n_grams.append('_'.join(words))
        return n_grams
