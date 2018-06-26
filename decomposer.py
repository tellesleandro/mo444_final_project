from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', \
                    level=logging.INFO, \
                    datefmt='%Y-%m-%d %H:%M:%S')

from pdb import set_trace as bp

class Decomposer:

    def __init__(self, dimensionality = 500):
        logging.info('decomposer.init')
        self.svd = TruncatedSVD(dimensionality)
        self.lsa = make_pipeline(self.svd, Normalizer(copy = False))

    def fit(self, X):
        logging.info('decomposer.fit')
        return self.lsa.fit_transform(X)

    def transform(self, X):
        logging.info('decomposer.transform')
        return self.lsa.transform(X)

    def explained_variance(self):
        logging.info('decomposer.explained_variance')
        return self.svd.explained_variance_ratio_.sum()
