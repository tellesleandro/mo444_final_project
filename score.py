from collections import Counter,defaultdict

import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', \
                    level=logging.INFO, \
                    datefmt='%Y-%m-%d %H:%M:%S')

from pdb import set_trace as bp

class Score:

    def __init__(self):
        logging.info('score.init')
        self.labels_count = {}
        self.labels_count = defaultdict(lambda: 0, self.labels_count)
        self.prediction_count = {}
        self.prediction_count = defaultdict(lambda: 0, self.prediction_count)
        self.labels_score = {}
        self.labels_score = defaultdict(lambda: 0, self.labels_score)
        self.normalized_score = 0

    def append_score(self, actual, prediction):
        logging.info('score.append_score')
        self.labels_count[actual] += 1
        if actual == prediction:
            self.prediction_count[actual] += 1
        else:
            self.prediction_count[actual] += 0

    def summarize(self):
        logging.info('score.summarize')
        for label, count in self.labels_count.items():
            self.labels_score[label] = self.prediction_count[label] / count
            self.normalized_score += self.labels_score[label]

        self.normalized_score /= len(self.labels_count)
