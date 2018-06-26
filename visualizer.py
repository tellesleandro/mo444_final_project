import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from pdb import set_trace as bp

class Visualizer:

    def __init__(self, X, y):
        self.__X = X
        self.__y = y
        self.tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        self.tsne_results = self.tsne.fit_transform(self.__X)

    def save_visualization(self):
        plt.figure(figsize=(18, 18))  # in inches
        for idx, x in enumerate(self.tsne_results):
            plt.scatter(x[0], x[1], label = self.__y[idx])
        plt.savefig('figure.png')
