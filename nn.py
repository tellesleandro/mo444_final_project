import numpy as np
np.set_printoptions(threshold = np.nan)
import datetime
import json

import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', \
                    level=logging.INFO, \
                    datefmt='%Y-%m-%d %H:%M:%S')

from pdb import set_trace as bp

class NN:

    def __init__(self, words, classes):
        logging.info('nn.init')
        self.words = words
        self.classes = classes

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y, \
                hidden_neurons = 10, \
                alpha = 1, \
                epochs = 50000, \
                dropout = False, \
                dropout_percent = 0.5):

        logging.info('nn.train')
        np.random.seed(1)

        last_mean_error = 1
        synapse_0 = 2 * np.random.random((len(X[0]), hidden_neurons)) - 1
        synapse_1 = 2 * np.random.random((hidden_neurons, len(self.classes))) - 1

        prev_synapse_0_weight_update = np.zeros_like(synapse_0)
        prev_synapse_1_weight_update = np.zeros_like(synapse_1)

        synapse_0_direction_count = np.zeros_like(synapse_0)
        synapse_1_direction_count = np.zeros_like(synapse_1)

        for j in iter(range(epochs + 1)):

            # Feed forward through layers 0, 1, and 2
            layer_0 = X
            layer_1 = self.sigmoid(np.dot(layer_0, synapse_0))

            if(dropout):
                layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

            layer_2 = self.sigmoid(np.dot(layer_1, synapse_1))

            # how much did we miss the target value?
            layer_2_error = y - layer_2

            if (j% 10000) == 0 and j > 5000:
                if np.mean(np.abs(layer_2_error)) < last_mean_error:
                    print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                    last_mean_error = np.mean(np.abs(layer_2_error))
                else:
                    print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                    break

            layer_2_delta = layer_2_error * self.sigmoid_derivative(layer_2)

            layer_1_error = layer_2_delta.dot(synapse_1.T)

            layer_1_delta = layer_1_error * self.sigmoid_derivative(layer_1)

            synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
            synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

            if(j > 0):
                synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
                synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))

            synapse_1 += alpha * synapse_1_weight_update
            synapse_0 += alpha * synapse_0_weight_update

            prev_synapse_0_weight_update = synapse_0_weight_update
            prev_synapse_1_weight_update = synapse_1_weight_update

        now = datetime.datetime.now()

        synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
                   'datetime': now.strftime("%Y-%m-%d %H:%M"),
                   'words': self.words,
                   'classes': self.classes
                  }
        synapse_file = "synapses.json"

        with open(synapse_file, 'w') as outfile:
            json.dump(synapse, outfile, indent=4, sort_keys=True)

    def classify(self, X):
        logging.info('nn.classify')
        ERROR_THRESHOLD = 0.2
        synapse_file = 'synapses.json'
        with open(synapse_file) as data_file:
            synapse = json.load(data_file)
            synapse_0 = np.asarray(synapse['synapse0'])
            synapse_1 = np.asarray(synapse['synapse1'])

        l0 = X
        l1 = self.sigmoid(np.dot(l0, synapse_0))
        l2 = self.sigmoid(np.dot(l1, synapse_1))

        results = l2

        results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ]
        results.sort(key=lambda x: x[1], reverse=True)
        return_results =[[self.classes[r[0]],r[1]] for r in results]
        print ("classification: %s" % (return_results))
        return return_results
