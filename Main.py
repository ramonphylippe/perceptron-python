from .core import perceptron
import numpy as np

Input = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
Output = np.array([[0, 1, 1, 1]])
solution = perceptron.perceptronWeightCalculation(100, 0.001, 0.5, Input, Output)

weight = solution[0]
bias = solution[1]

sim = perceptron.sim(weight, bias, Input)

print("Saida do algoritimo: ", sim)