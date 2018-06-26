from sklearn.naive_bayes import MultinomialNB

import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', \
                    level=logging.INFO, \
                    datefmt='%Y-%m-%d %H:%M:%S')

from pdb import set_trace as bp

class NaiveBayesMultinomial:

    def fit(self, X, y):
        logging.info('naive_bayes_multinomial.fit')
        self.classifier = MultinomialNB()
        self.classifier.fit(X, y)

    def predict(self, X):
        logging.info('naive_bayes_multinomial.predict')
        prediction = self.classifier.predict(X)
        return prediction
