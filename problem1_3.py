import numpy as np
import pandas as pd
import csv
import sys

class PerceptronLearner():

    def __init__(self):
        self.inputFile = None
        self.ouputFile = None

    def read_data(self):
        data = pd.read_csv(self.inputFile)
        data = data.values
        return data

    def sign(self, row, weights):
        firstValue = row[0] * weights[0]
        secondValue = row[1] * weights[1]
        sum = weights[2] + firstValue + secondValue
        return 1 if sum >= 0 else -1


    def csvWriter(self, row):
        with open('output1.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            writer.writerow(row)

    def perceptronAlgorithm(self):
        data  =  self.read_data()
        weights = [0 for i in range(len(data[0]))]
        result = ""
        while True:
            isFinal = True
            for i in range(0, len(data)):
                expected = data[i][2]
                predicted = self.sign(data[i], weights)
                if expected * predicted <= 0:
                    isFinal = False
                    weights[0] = weights[0] + expected * data[i][0]
                    weights[1] = weights[1] + expected * data[i][1]
                    weights[2] = weights[2] + expected

            if isFinal:
                result += str(weights[0]) + ", " + str(weights[1]) + ", " + str(weights[2])
                break
            else:
                result += str(weights[0]) + ", " + str(weights[1]) + ", " + str(weights[2]) + "\n"
            self.csvWriter([weights[0], weights[1], weights[2]])



if __name__ == '__main__':
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    result = PerceptronLearner(inputfile, outputfile)
    result.perceptronAlgorithm()