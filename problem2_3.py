import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import csv
class LinearRegression:

    def __init__(self, inputfilep, outputfilep):
        self.inputFile = inputfilep
        self.outputFile = outputfilep
        self.learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.003]
        self.iters = [100, 100, 100, 100, 100, 100, 100, 100, 100, 500]


    def dataPrep(self):
        data = pd.read_csv(self.inputFile)
        X = data.iloc[:, 0:2].values
        y = data.iloc[:, 2].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        samples = len(y)
        features = np.ones(shape=(samples, 3))

        features[:, 1:3] = X
        return features, y

    def csvWriter(self, row):
        with open('output2.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            writer.writerow(row)

    def cost_func(self, features, labels, beta):

        return np.sum((features.dot(beta) - labels) ** 2) /2 / len(labels)

    def regressor(self):
        features, labels = self.dataPrep()
        samples = len(labels)

        for iteration, rate in zip(self.iters, self.learning_rates):
            betas = np.array([0, 0, 0])
            history = []
            for iterate in range(iteration):
                prediction = features.dot(betas)
                loss = prediction - labels
                gradient = features.T.dot(loss)/samples
                betas = betas - rate * gradient
                print(betas)
                history.append(self.cost_func(features, labels, betas))
            self.csvWriter([rate, iteration, betas[0], betas[1], betas[2]])



if __name__=='__main__':
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    regression = LinearRegression(inputfile, outputfile)
    regression.regressor()
