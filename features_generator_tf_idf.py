import os
import re
import html
from math import log
from collections import Counter,defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import pickle

import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', \
                    level=logging.INFO, \
                    datefmt='%Y-%m-%d %H:%M:%S')

from pdb import set_trace as bp

class FeaturesGeneratorTfIdf:

    STRIP_SEQUENCE = "([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})|[!#$%^&*()\_\-–+={}\[\]\\\:;'<,>\.?/`´“”²³\|" + '"]'

    def __init__(self):
        self.__vectorizer = TfidfVectorizer()

    def fit(self, texts, labels):
        logging.info('features_generator.fit')
        corpus, labels = self.clean_corpus(texts, labels)
        features = self.__vectorizer.fit_transform(corpus)
        return features, labels

    def transform(self, texts, labels):
        logging.info('features_generator.transform')
        corpus, labels = self.clean_corpus(texts, labels)
        features = self.__vectorizer.transform(corpus)
        return features, labels

    def clean_corpus(self, corpus, labels):
        logging.info('features_generator.clean_corpus')
        new_corpus = []
        new_labels = []
        for idx, document in corpus.items():
            document = self.clean_text(document)
            if len(document) == 0: continue
            new_corpus.append(document)
            new_labels.append(labels[idx])
        return new_corpus, new_labels

    def clean_text(self, text):
        logging.info('features_generator.clean_text')
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
        logging.info('features_generator.remove_tags')
        text = re.sub('<.*?>', '', text)
        text = re.sub('\s+', ' ', text ).strip()
        return text

    def vocabulary(self):
        return self.__vectorizer.vocabulary_

    def save_engine_features_and_labels(self, datadir):
        logging.info('features_generator.save_engine_features_and_labels')
        dirname = os.path.dirname(datadir)
        vectorizer_filename = dirname + '/vectorizer.pkl'
        x_filename = dirname + '/features.npz'
        y_filename = dirname + '/labels.pkl'
        with open(vectorizer_filename, 'wb') as fp:
            pickle.dump(self.__vectorizer, fp)
        sparse.save_npz(x_filename, self.X)
        with open(y_filename, 'wb') as fp:
            pickle.dump(self.y, fp)

    def load_engine_features_and_labels(self, datadir):
        logging.info('features_generator.load_engine_features_and_labels')
        dirname = os.path.dirname(datadir)
        vectorizer_filename = dirname + '/vectorizer.pkl'
        x_filename = dirname + '/features.npz'
        y_filename = dirname + '/labels.pkl'
        try:
            with open(vectorizer_filename, 'rb') as fp:
                self.__vectorizer = pickle.load(fp)
            self.X = sparse.load_npz(x_filename)
            with open(y_filename, 'rb') as fp:
                self.y = pickle.load(fp)
            return True
        except:
            return False
