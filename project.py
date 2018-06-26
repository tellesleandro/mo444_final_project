from dataset_handler import *
from sklearn.model_selection import KFold
from features_generator_bow import *
from nn import *
from math import sqrt
from decomposer import *
from visualizer import *
from classifier import *
from statistics import *
from score import *

import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', \
                    level=logging.INFO, \
                    datefmt='%Y-%m-%d %H:%M:%S')

from pdb import set_trace as bp

ds = DatasetHandler('dataset/dataset.csv')
ds.load()
ds.clean()
ds.save()

ds = DatasetHandler('dataset/dataset_handled.csv')
ds.load()
ds.plot_classes_frequency()

texts = ds.dataset.descricao
labels = ds.dataset.area

k_fold = KFold()

for train_indexes, validation_indexes in k_fold.split(texts):

    X_train, y_train = texts[train_indexes], labels[train_indexes]
    X_validation, y_validation = texts[validation_indexes], labels[validation_indexes]

    fg = FeaturesGeneratorBoW()
    fg.build_model(X_train, y_train)
    fg.generate_features_and_targets_for_model()

    X = fg.model_features
    dc = Decomposer()
    X = dc.fit(X)
    y = fg.model_targets

    nn = NN(fg.words, fg.classes)
    hidden_neurons = int(len(X[0]) / 2)
    print('# of features:', len(X[0]), '(', dc.explained_variance() ,')' ';', 'Hidden neurons:', hidden_neurons)

    nn.train(X, y, \
            hidden_neurons = hidden_neurons, \
            alpha = 0.1, \
            epochs = 100000, \
            dropout = False, \
            dropout_percent = 0.2)

    score = Score()

    for idx, text in enumerate(X_validation):
        print(text)
        bow = fg.bow_from_text(text)
        bow = dc.transform([bow])
        prediction = nn.classify(bow[0])
        prediction = prediction[0][0] if len(prediction) > 0 else ''
        score.append_score(y_validation[idx], prediction)
        print(y_validation[idx], '|', prediction)
        print()

    score.summarize()

    for label, label_score in score.labels_score.items():
        print(label, '|', score.labels_count[label], '|', score.prediction_count[label], '|', label_score)

    print('Normalized score:', score.normalized_score)
