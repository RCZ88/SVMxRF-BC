import math
import time
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from cvxopt import matrix, solvers
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from svmOBJ import *
from ResultData import ResultData
from TuningDataStatistics import *
from index import *
from tools import *

filename = 'Breast Cancer Wisconsin DB'
data = pd.read_csv(f"C:\\Users\\ASUS\\PycharmProjects\\MachineLearning\\FINAL SVM & RF\\{filename}.csv")
# id has no purpose on machine learning
data = data.drop('id', axis=1)
data = data.drop('Unnamed: 32', axis=1)
finalColIndex = 0
sampleRowCount: int = 400

data = data.sample(n=sampleRowCount)

# mapping the final col from letters to numbers
diagnosisReference = {'M': -1, 'B': 1}
data['diagnosis'] = data['diagnosis'].map(diagnosisReference)

data = data.to_numpy()

def get_x(data, finalColIndex):
    column_no_y = [i for i in range(data.shape[1]) if i != finalColIndex]
    return data[:, column_no_y]

def get_y(data, finalColIndex):
    arr = np.asarray(data)
    if arr.ndim == 1:
        return data[finalColIndex]
    return data[:, finalColIndex]

def get_k_matrix(xdata, kernel, c= None, d= None, gamma= None):
    rowAmount = xdata.shape[0]
    columnAmount = xdata.shape[1]
    kmatrix = np.zeros((rowAmount, rowAmount))
    for i in range(rowAmount):
        for j in range(rowAmount):
            sumRow = 0
            if kernel == 'rbf':
                diffSquared = 0
                for k in range(columnAmount):
                    diff =  xdata[i, k] - xdata[j, k]
                    diffSquared += diff**2
                sumRow += np.exp(-gamma * diffSquared)
            elif kernel == 'polynomial':
                for k in range(columnAmount):
                    sumRow += (xdata[i, k] * xdata[j, k] + c) ** d
            elif kernel == 'linear':
                for k in range(columnAmount):
                    sumRow += xdata[i, k] * xdata[j, k]
            kmatrix[i, j] = sumRow
    return kmatrix

def get_y_matrix(ydata):
    return ydata.reshape(1, -1)

def get_q_matrix(k_matrix, ydata):
    size = k_matrix.shape[0]
    qmatrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            qmatrix[i, j] = k_matrix[i, j] * ydata[i] * ydata[j]
    return qmatrix

def calculate_a(data, matrix_size, kernel, C=1, c= None, d= None, gamma= None):
    y = get_y(data, finalColIndex)
    x = get_x(data, finalColIndex)
    k = get_k_matrix(x, kernel, c = c, d = d, gamma = gamma)
    q = matrix(get_q_matrix(k, y))
    p = matrix([-1.0] * matrix_size)
    g = matrix(np.vstack([-np.eye(matrix_size), np.eye(matrix_size)]))
    h = matrix(np.hstack([np.zeros(matrix_size), C * np.ones(matrix_size)]))
    a = matrix(get_y_matrix(y))
    b = matrix([0.0])
    #Qm, pm, Gm, hm, Am, bm
    solvers.options['show_progress'] = False
    solution = solvers.qp(q, p, g, h, a, b)
    return np.array(solution['x']).flatten()

#since non-linear uses the xi which is the existent training data, and xj being the support vector selected row
def get_k_for_nonlinear(xdata, xj, kernel, c= None, d= None, gamma= None):
    size = xdata.shape[0]
    columnAmount = xdata.shape[1]
    #creates a 1D horizontal array to store the values
    krow = np.zeros(size)
    #goes to every row -> length of equation directly proportional with the row/size of data
    for i in range(size):
        sumRow = 0
        if kernel == 'rbf':
            diffSquared = 0
            for k in range(columnAmount):
                diff = xdata[i, k] - xj[k]
                diffSquared += diff ** 2
            sumRow += np.exp(-gamma * diffSquared)
        elif kernel == 'polynomial':
            for k in range(columnAmount):
                sumRow += (xdata[i, k] * xj[k] + c) ** d
        elif kernel == 'linear':
            for k in range(columnAmount):
                sumRow += xdata[i, k] * xj[k]
        krow[i] = sumRow
    #returns a vertical instead of a horizontal
    return krow.reshape(-1, 1)

###multiply the list of alphas, x, y to get the "w" from the equation y = w^T x+b. Getting the value to find b
def get_w(alphas, x, y):
    return np.sum((alphas * y)[:, None] * x, axis=0)

#only for linear
def get_b(alphas, x, y, w):
    svi = get_support_vectors(alphas)
    xsv = x[svi] #support vector of x
    ysv = y[svi] #support vector of y
    wx = np.dot(xsv, w)
    b_values = ysv - np.sum(wx)
    b = np.mean(b_values)
    return b

def get_support_vectors(alphas):
    threshold = 1e-5
    support_vector_indices = np.where(alphas > threshold)[0]
    return support_vector_indices

def do_linear(data, finalColIndex, C):
    size = data.shape[0]
    x = get_x(data, finalColIndex)
    y = get_y(data, finalColIndex)
    alphas = calculate_a(data, x.shape[0], "linear", C = C)
    w = get_w(alphas, x, y)
    b = get_b(alphas, x, y, w)
    return Linear(size, C, w, b)

def do_polynomial(data, finalColIndex, C, c, d):
    size = data.shape[0]
    x = get_x(data, finalColIndex)
    y = get_y(data, finalColIndex)
    alphas = calculate_a(data, x.shape[0], 'polynomial', c = c, d = d, C = C)
    sv = get_support_vectors(alphas)
    xj = x[sv[0]]
    krow = get_k_for_nonlinear(x, xj, 'polynomial', c = c, d = c)
    sumvalues = krow * y * alphas
    sum = np.sum(sumvalues)
    #since -b = sum
    b = -sum
    return Polynomial(alphas, C, y, b, x, size, c, d)

def do_rbf(data, finalColIndex, C, gamma):
    size = data.shape[0]
    x = get_x(data, finalColIndex)
    if gamma == -1.0:
        sd = np.std(x, ddof=1)
        gamma = 1/(2*(sd**2))
    y = get_y(data, finalColIndex)
    alphas = calculate_a(data, x.shape[0], 'rbf', C = C, gamma = gamma)
    sv = get_support_vectors(alphas)
    xj = x[sv[0]]
    krow = get_k_for_nonlinear(x, xj, 'rbf', gamma = gamma)
    sumvalues = krow * y * alphas
    sum = np.sum(sumvalues)
    # since -b = sum
    b = -sum
    return RBF(alphas, C, y, b, x, size, gamma)

def cross_validation(data, kernel, finalColIndex, n_components = 10, kFold=5, C=1, c= None, d= None, gamma= None):
    indices = np.arange(data.shape[0])
    np.random.seed(42)
    np.random.shuffle(indices)
    data = data[indices, :]
    trainingData = []
    testData = []
    chunkSize = data.shape[0] // kFold
    for i in range(kFold):
        trainDataSeperated = []
        for j in range(kFold):
            if j == i:
                testData.append(data[j * chunkSize:(j + 1) * chunkSize])
            else:
                trainDataSeperated.append(data[j * chunkSize:(j + 1) * chunkSize])
        trainingData.append(np.vstack(trainDataSeperated))

    for i in range(kFold):
        X_train = get_x(trainingData[i], finalColIndex)
        y_train = get_y(trainingData[i], finalColIndex)
        X_test = get_x(testData[i], finalColIndex)
        y_test = get_y(testData[i], finalColIndex)

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        pca = PCA(n_components=n_components).fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

        trainingData[i] = np.hstack([y_train.reshape(-1, 1), X_train])
        testData[i] = np.hstack([y_test.reshape(-1, 1), X_test])

    confusionMatrix = getConfusionMatrix()
    kf_trainTime_avg = 0;
    kf_testTime_avg = 0;
    for k in range(kFold):
        timeStart = time.time()
        svmObj = None
        if kernel == 'linear':
            svmObj:Linear = do_linear(trainingData[k], finalColIndex, C)
        elif kernel == 'polynomial':
            svmObj:Polynomial = do_polynomial(trainingData[k], finalColIndex, C, c, d)
        elif kernel == 'rbf':
            svmObj:RBF = do_rbf(trainingData[k], finalColIndex, C, gamma)
        timeEnd = time.time()
        trainTime = timeEnd - timeStart
        kf_trainTime_avg += trainTime
        true = 0
        false = 0
        useTime = 0
        for i in range(testData[k].shape[0]):
            timeStart = time.time()
            calculationResult = svmObj.calculate(get_x(testData[k][i, :].reshape(1, -1), finalColIndex))
            timeEnd = time.time()
            useTime += (timeEnd - timeStart)
            if calculationResult is not None:
                response = -1
                if calculationResult > 0:
                    response = 1
                if get_y(testData[k][i], finalColIndex) == response:
                    if response == 1:
                        confusionMatrix['TP'] += 1
                    elif response == -1:
                        confusionMatrix['TN'] += 1
                else:
                    if response == 1:
                        confusionMatrix['FP'] += 1
                    elif response == -1:
                        confusionMatrix['FN'] += 1
        avgUseTime = useTime / testData[k].shape[0]
        kf_testTime_avg += avgUseTime
    kf_trainTime_avg /= kFold
    kf_testTime_avg /= kFold
    print(f'Train Time Average(of k folds): {kf_trainTime_avg}')
    print(f'Test Time Average(of k folds): {kf_testTime_avg}')
    return confusionMatrix



def hyperparameter_tuning(hyperparameters, data, finalColIndex):
    for typeOfSVM in hyperparameters['Types Of SVM: ']:
        if typeOfSVM == 'Linear':
            for C in hyperparameters['C Values: ']:
                print(f'Testing for Hyperparameter: Linear C:({C})')
                cm = cross_validation(data, "linear", finalColIndex, C = C)
                print(f'CM: {cm}')
                print(f'Metrics: {get_metrics(cm)}\n')
        if typeOfSVM == 'RBF':
            for C in hyperparameters['C Values: ']:
                for gamma in hyperparameters['RBF Gamma Values: ']:
                    print(f'Testing for Hyperparameter: RBF C:({C}), gamma:({gamma})')
                    cm = cross_validation(data, "rbf", finalColIndex, C = C, gamma = gamma)
                    print(f'CM: {cm}')
                    print(f'Metrics: {get_metrics(cm)}\n')
        if typeOfSVM == 'Polynomial':
            for C in hyperparameters['C Values: ']:
                for c in hyperparameters['Polynomial C Values: ']:
                    for d in hyperparameters['Polynomial D Values: ']:
                        print(f'Testing for Hyperparameter: Polynomial C:({C}), c:({c}), d:({d})')
                        cm = cross_validation(data, 'polynomial', finalColIndex, C = C, c  = c, d = d)
                        print(f'CM: {cm}')
                        print(f'Metrics: {get_metrics(cm)}\n')

def get_metrics(confusionMatrix: dict[str:int]):
    TP = confusionMatrix['TP']
    TN = confusionMatrix['TN']
    FP = confusionMatrix['FP']
    FN = confusionMatrix['FN']
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)
    return {'ACCURACY':accuracy, 'PRECISION':precision, 'RECALL':recall, 'F1SCORE':f1_score}

def getConfusionMatrix():
    return {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

hyperparameters = {'Types Of SVM: ':['Polynomial'],
                   'C Values: ':[0.1, 1, 10, 100, 1000], #5
                   'RBF Gamma Values: ':[-1.0, 0.001, 0.01, 0.1, 1, 10, 100], #5 * 7 = 35
                   'Polynomial C Values: ':[i for i in range(1, 5)],
                   'Polynomial D Values: ': [2, 3, 4]} # 5 * 4 * 3 = 60

startTime = time.time()
hyperparameter_tuning(hyperparameters, data, finalColIndex)
endTime = time.time()
timeTaken = endTime - startTime
print(f'Time Taken: {timeTaken}')