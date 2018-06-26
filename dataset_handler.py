import os
import re
import html
import pandas
import matplotlib.pyplot as plt

import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', \
                    level=logging.INFO, \
                    datefmt='%Y-%m-%d %H:%M:%S')

from pdb import set_trace as bp

class DatasetHandler:

    def __init__(self, filename):
        logging.info('dataset_handler.init')
        self.__filename = filename

    def load(self):
        logging.info('dataset_handler.load')
        self.dataset = pandas.read_csv(self.__filename)

    def clean(self):
        logging.info('dataset_handler.clean')
        self.dataset['descricao'] = \
            self.dataset.descricao.apply(self.clean_descricao)

    def clean_descricao(self, descricao):

        if not isinstance(descricao, str) or descricao is None or len(descricao) == 0:
            return ''

        previous_descricao = descricao

        while True:
            actual_descricao = html.unescape(previous_descricao)
            if actual_descricao == previous_descricao: break
            previous_descricao = actual_descricao

        while True:
            actual_descricao = self.remove_tags(previous_descricao)
            if actual_descricao == previous_descricao: break
            previous_descricao = actual_descricao

        return previous_descricao

    def remove_tags(self, text):
        text = re.sub('<.*?>', '', text)
        text = re.sub('\s+', ' ', text ).strip()
        return text

    def plot_classes_frequency(self):
        self.dataset.area.value_counts().plot(kind="bar", rot=0)
        plt.xticks(rotation=90)
        plt.show()

    def save(self):
        logging.info('dataset_handler.save')
        dirname = os.path.dirname(self.__filename)
        filename = os.path.basename(self.__filename)
        base, ext = os.path.splitext(filename)
        handled_filename = dirname + '/' + base + '_handled' + ext
        self.dataset.to_csv(handled_filename, index = False)
