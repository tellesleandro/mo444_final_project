from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', \
                    level=logging.INFO, \
                    datefmt='%Y-%m-%d %H:%M:%S')

from pdb import set_trace as bp

class Classifier:

    def __init__(self, X, y):
        logging.info('classifier.init')
        n_neighbors = len(Counter(y))
        self.knn = KNeighborsClassifier( \
                                    n_neighbors = n_neighbors, \
                                    )
        self.knn.fit(X, y)

    def predict(self, X):
        logging.info('classifier.predict')
        return self.knn.predict(X)
