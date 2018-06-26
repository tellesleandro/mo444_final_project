import re
import html
from math import log
from collections import Counter,defaultdict
import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', \
                    level=logging.INFO, \
                    datefmt='%Y-%m-%d %H:%M:%S')

from pdb import set_trace as bp

class NaiveBayes:

    STRIP_SEQUENCE = "([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})|[!#$%^&*()\_\-+={}\[\]\\\:;'<,>\.?/`´“”²³\|" + '"]'

    def train(self, texts, labels):
        logging.info('statistical.train')

        self.labels = Counter(labels)
        self.priors = {}
        self.priors = defaultdict(lambda: 0, self.priors)

        for label, count in self.labels.items():
            self.priors[label] = log(count / len(labels))

        self.dict = []
        self.word_counts_by_class = {}

        for idx, text in texts.items():

            if not isinstance(text, str) or text is None or len(text) == 0:
                continue

            label = labels[idx]

            if label not in self.word_counts_by_class:
                self.word_counts_by_class[label] = {}

            text = self.clean_text(text)
            word_counts = self.count_words(text)

            for word, count in word_counts.items():
                if word not in self.dict: self.dict.append(word)
                if word not in self.word_counts_by_class[label]:
                    self.word_counts_by_class[label][word] = 0.0
                self.word_counts_by_class[label][word] += count

        for label, words in self.word_counts_by_class.items():
            print(label, '|')
            sorted_words = sorted(words.items(), key=lambda x: x[1])
            for word, count in sorted_words:
                print('|', word, '|', count)

        import sys
        sys.exit(0)

    def validate(self, texts, labels):
        logging.info('statistical.validate')
        prediction_count = {}
        prediction_count = defaultdict(lambda: 0, prediction_count)
        for idx, text in texts.items():
            if not isinstance(text, str) or text is None or len(text) == 0:
                continue
            actual = labels[idx]
            prediction = self.predict(text)
            print(idx, text, '[' + actual + ']', '[' + prediction + ']')
            print()
            if actual == prediction:
                prediction_count[actual] += 1
            else:
                prediction_count[actual] += 0

        normalized_score = 0
        labels_count = Counter(labels)
        for label, count in labels_count.items():
            label_score = prediction_count[label] / count
            print(label, count, prediction_count[label], label_score)
            normalized_score += label_score

        normalized_score /= len(labels_count)
        print(normalized_score)

    def predict(self, text):
        logging.info('naive_bayes.predict')
        labels_scores = {}
        labels_scores = defaultdict(lambda: 0, labels_scores)
        text = self.clean_text(text)
        word_counts = self.count_words(text)
        for word, count in word_counts.items():
            if word not in self.dict: continue
            for label, count in self.labels.items():
                label_score = log( \
                                (self.word_counts_by_class[label].get(word, 0.0) + 1) / \
                                (sum(self.word_counts_by_class[label].values()) + len(self.dict)) \
                                )
                labels_scores[label] += label_score

        for label, score in self.labels.items():
            labels_scores[label] += self.priors[label]

        return sorted(labels_scores, key = labels_scores.get)[-1]

    def clean_text(self, text):
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

        actual_text = re.sub(self.STRIP_SEQUENCE, " ", actual_text.lower()).strip()
        actual_text_valid_words = []
        for word in actual_text.split():
            if (len(word) > 3 and len(word) < 20) and not word.isdigit():
                actual_text_valid_words.append(word)

        actual_text = ' '.join(actual_text_valid_words)

        return actual_text

    def remove_tags(self, text):
        text = re.sub('<.*?>', '', text)
        text = re.sub('\s+', ' ', text ).strip()
        return text

    def count_words(self, text):
        word_counts = {}
        word_counts = defaultdict(lambda: 0, word_counts)
        for word in text.split():
            word_counts[word] += 1
        return word_counts

    def slice_by_classes(self):
        logging.info('statistical.slice_by_classes')
        self.dataset_slices = {}
        classes = self.get_classes()
        for klass in classes:
            self.dataset_slices[klass] = self.rows_for_class(klass)

    def get_classes(self):
        logging.info('statistical.classes')
        if not hasattr(self, 'classes'):
            self.classes = self.dataset.area.unique()
        return self.classes

    def rows_for_class(self, klass):
        logging.info('statistical.rows_for_class[' + klass + ']')
        return self.dataset.loc[self.dataset.area == klass]

    def priors(self):
        logging.info('statistical.priors')
        self.priors = {}
        for klass, slice in self.dataset_slices.items():
            self.priors[klass] = log(len(slice) / len(self.dataset))

    def dictionary(self):
        logging.info('statistical.dictionary')
        self.dict = []
        self.word_counts_by_class = {}
        for idx, row in self.dataset.iterrows():
            klass = row.area
            if klass not in self.word_counts_by_class:
                self.word_counts_by_class[klass] = {}
            sentence = row.descricao
            word_counts = self.count_words(sentence)
            for word, count in word_counts.items():
                if word not in self.dict: self.dict.append(word)
                if word not in self.word_counts_by_class[klass]:
                    self.word_counts_by_class[klass][word] = 0.0
                self.word_counts_by_class[klass][word] += count
