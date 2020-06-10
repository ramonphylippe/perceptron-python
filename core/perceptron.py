import random
import numpy as np


#função tretosa pra funcionar
def perceptronWeightCalculation(MaxIteration, MinErrorCriteria, WeigthForTuning, InputData, OutputData):
    t = 1
    Error = 100000
    getNumberOfRows = np.shape(InputData)[0]
    getNumberOfColumns = np.shape(InputData)[1]
    weight = np.array([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)])
    bias = 2*(random.uniform(1.0, 5))

    print("\nPesos:\n", weight)
    print("Bias:\n", bias)
    print("Dados de entrada:\n", InputData)
    print("Dados de saida:\n", OutputData[0])
    print("*****************************\n")

    while (t < MaxIteration) and (Error > MinErrorCriteria):
        y = np.zeros(getNumberOfColumns)
        e = np.zeros(getNumberOfColumns)

        for iColumns in range(getNumberOfColumns):

            tOutput = 0
            for kRows in range(getNumberOfRows):
                tOutput += weight[kRows] * InputData[kRows, iColumns]

            y[iColumns] = HardLimit(tOutput+bias)

            e[iColumns] = OutputData[0, iColumns]-y[iColumns]

            for kRows in range(getNumberOfRows):
                weight[kRows] += WeigthForTuning*e[iColumns] * InputData[kRows, iColumns]
                # print("linha 36", weight[kRows])
            bias += WeigthForTuning*e[iColumns]

            print("Erros\n", e)
            print("Pesos\n", weight)
            print("bias\n", bias)

        err = sum(e+1-1)
        Error = sum(abs(e))
        ++t

    return weight, bias


#Função de ativação
def HardLimit(value):
    if value < 0.0:
        return 0
    else:
        return 1


#função para gerar valores dos pesos
def sim(weight, bias, InputData):
    output = np.mat(weight) * np.mat(InputData) + bias
    print("\n\n******************************************************")
    print("Saida: ", output)
    getNumberOfColumns = np.shape(output)[1]
    for iColumns in range(getNumberOfColumns):
        output[0, iColumns] = HardLimit(output[0, iColumns])
    return output
