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

scaleData = True
doPca = True
methods = ['cv', 'noncv']

# Load data
dataFileName = 'Breast Cancer Wisconsin DB'
data = pd.read_csv(f'C:\\Users\\ASUS\\PycharmProjects\\MachineLearning\\FINAL SVM & RF\\{dataFileName}.csv')
outcomeReference = {'M': -1, 'B': 1}
data['diagnosis'] = data['diagnosis'].map(outcomeReference)

n = 400#sample size

print(data.shape[1])
x_indexes = [i for i in range(2, data.shape[1]-1)]  # Indices of feature columns
print(x_indexes)
y_index = 1       # Index of label column
sampleData = data.iloc[:, x_indexes + [y_index]].copy()
sampleData = sampleData.sample(n=n, random_state=42)
sampleFileName = f'SD{dataFileName}SS{n}X{len(x_indexes)}'
varCount = len(x_indexes)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(sampleData.iloc[:, :-1].values)

sampleFileName += 'SCALED'

pcaComponents = 10
pca = PCA(n_components=pcaComponents)
X_pca = pca.fit_transform(X_scaled)
print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")

pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
x_indexes = [i for i in range(1, X_pca.shape[1]+1)]
sampleData_pca = pd.DataFrame(X_pca, columns=pca_columns)
sampleData_pca['diagnosis'] = sampleData['diagnosis'].values
sampleData = sampleData_pca  # Replace original data with PCA-transformed data
varCount = pcaComponents

sampleFileName += f'PCA{pcaComponents}'

sampleFileName = sampleFileName + '.csv'
sampleFileName = sampleFileName.replace(" ", "")
# print(type(sampleData))

sampleData.to_csv(sampleFileName, index=False)
print(sampleData)


# Prepare matrices
# matrixSize = sampleData.shape[0]  # Number of rows in the dataset


sampleX = [i for i in range(varCount)]
sampleY = [varCount]
def getX(data):
    x = data.iloc[:, :-1].values
    return x


def getY(data, yIndex=-1):
    if yIndex == -1:
        y = data.iloc[:, yIndex].values
        return y
    y = data.iloc[:, [yIndex]].values
    return y
# print("X shape:", x.shape)  # Should be (n, num_features)
# print("y shape:", y.shape)  # Should be (n,)



def calculate_avg_distance(x):
    distanceTotal = 0
    count = 0
    for i in range(len(x)):
        for j in range(len(x)):
            sumVars = 0
            for k in range(varCount):
                sumVars += (x[i, k] - x[j, k]) ** 2
            distanceTotal += math.sqrt(sumVars)
            count += 1
    return distanceTotal / count



def linear_kernel(x, matrixSize):
    K = np.zeros((matrixSize, matrixSize)) # Kernel matrix (dot products)
    for row in range(len(K)):
        for col in range(len(K[row])):
            K[row][col] = np.sum(x[col] * x[row])
    return K

def polynomial_kernel(matrixSize, x, c = 1, d = 2):
    K = np.zeros((matrixSize, matrixSize))
    for row in range(len(K)):
        for col in range(len(K[row])):
            K[row][col] = calc_polynomial(c, d, x[col], x[row])
    return K


def calc_polynomial(c, d, col, row):
    sumVar = np.sum(col * row)
    return (sumVar + c) ** d


def RBF_matrixQ(matrixSize, x, gamma = 1/ varCount):
    K = np.zeros((matrixSize, matrixSize))
    for i in range(len(K)):
        for j in range(len(K[i])):
            K[i][j] = calcRBF(gamma, x[i], x[j])
    return K

def calcRBF(gamma, xi, xj):
    sumVar = np.sum((xi - xj)**2)
    return np.exp(-gamma * sumVar)

def sigmoid_kernel(matrixSize, x, a, c):
    print('x1: ' , x[1])
    K = np.zeros((matrixSize, matrixSize))
    for i in range(len(K)):
        for j in range(len(K[i])):
            K[i][j] = calc_sigmoid(x[i], x[j], a, c)
    return K

def calc_sigmoid(i, j, a ,c):
    sumVar = np.sum(i * j)
    return np.tanh(a *sumVar + c)


def format_scientific(val):
    """Formats numbers in standard or scientific notation as needed."""
    if isinstance(val, np.ndarray):  # Ensure val is a scalar
        val = val.item()

    if val == 0:
        return "0"

    formatted = "{:.6e}".format(val)  # Convert to scientific notation
    base, exponent = formatted.split("e")  # Split base and exponent
    base = float(base)  # Convert back to float to avoid issues
    exponent = int(exponent)  # Convert exponent to integer

    # Only use scientific notation if exponent is significant
    if abs(exponent) >= 2:
        return f"{base:.2f} × 10^{{{exponent}}}"
    else:
        return f"{base:.2f}"  # Use standard decimal format

def get_a(matrixSize, Q, y, C=1.0):
    p = [-1 for _ in range(matrixSize)]  # Linear coefficient vector
    C = C  # Soft-margin parameter
    G = np.vstack([-np.eye(matrixSize), np.eye(matrixSize)])  # α ≥ 0, α ≤ C
    h = np.hstack([np.zeros(matrixSize), C * np.ones(matrixSize)])
    A = y.reshape(1, -1)   # Row vector of labels for equality constraint
    b = np.array([0.0])  # Scalar for equality constraint
    # Convert inputs to CVXOPT matrix format
    Qm = matrix(Q, tc='d')  # Q must be a symmetric matrix of type 'd'
    pm = matrix(np.array(p, dtype='d').reshape(-1, 1))  # p must be a column vector of type 'd'
    Gm = matrix(G.astype('d'))  # Ensure it's the correct type
    hm = matrix(h.astype('d'))  # Ensure 'h' is also properly formatted
    Am = matrix(A, tc='d')  # A must be a row vector of type 'd'
    bm = matrix(b, tc='d')  # b must be a scalar or column vector of type 'd'

    print("Am shape:", Am.size)  # This should be (1, matrixSize)
    print("Qm shape:", Qm.size)  # This should be (matrixSize, matrixSize)
    print("y shape:", y.shape)  # This should be (matrixSize, 1)

    # Solve the QP problem
    solvers.options['show_progress'] = False
    solution = solvers.qp(Qm, pm, Gm, hm, Am, bm)
    return solution


def getQMatrix(matrixSize, y, K):
    Q = np.zeros((matrixSize, matrixSize))  # Scaled kernel matrix
    for row in range(len(Q)):
        for col in range(len(Q[row])):
            Q[row][col] = y[row] * y[col] * K[row][col]
    return Q


def getSupportVectorRow(alphas):
    support_vectors = alphas > 1e-5  # Small threshold for numerical precision

    # print("Support Vector Indices:", np.where(support_vectors)[0])
    print("Number of Support Vectors:", np.sum(support_vectors))

    sv, index = None, -1
    for i, row in enumerate(sampleData.itertuples(index=False)):
        if support_vectors[i]:
            sv, index = row, i
            return index

def getAvgDot(matrixSize, x):
    avgDot = 1/(matrixSize**2)
    sum = 0
    for i in range(matrixSize):
        for j in range(matrixSize):
            sum+= abs(np.sum(x[i] * x[j]))
    return avgDot * sum

def findMargin(yk, alphas):
    sumQA = 0
    for i in range(len(yk)):
        for j in range(len(yk[i])):
            sumQA += yk[i][j] * alphas[i] * alphas[j]

    print(f'sumQA: {sumQA}')
    margin = 2 / (sumQA ** 0.5)
    print(f'Margin: {margin}')


def do_RBF(data, x, y, matrixSize, gamma = -1.0, pP = False, C= 1):

    avgDis = calculate_avg_distance(x)
    if gamma == -1.0:
        gamma = 1/(2*(avgDis**2))
    if pP:
        print(f'Average Distance: {avgDis}')
        print(f'Gamma: {gamma}')


    ynk = np.zeros((matrixSize, matrixSize))
    for i in range(matrixSize):
        for j in range(matrixSize):
            ynk[i][j] = y[i] * y[j] * calcRBF(gamma, x[i], x[j])


    K = RBF_matrixQ(matrixSize, x, gamma)

    Q = getQMatrix(matrixSize, y, K)

    solution = get_a(matrixSize, Q, y, C = C)

    # Extract the Lagrange multipliers (α)
    alphas = np.array(solution['x']).flatten()
    if pP:
        print("Lagrange Multipliers (α):", alphas)
    findMargin(ynk, alphas)

    svIndex = getSupportVectorRow(alphas)
    yk = y[svIndex]
    xk = x[svIndex]

    b = yk
    sumK = 0
    for i in range(x.shape[0]):
        sumK += alphas[i] * y[i] * calcRBF(gamma, x[i], xk)

    b -= sumK
    if isinstance(b, np.ndarray) and b.size == 1:
        b = b.item()  # Extracts a single number from the array
    d = RBF(alphas, C, y, b, x, matrixSize, gamma)
    if pP:
        print(f'b: {b}')
    equation = ''
    if varCount <= 3:
        xyz = ['x', 'y', 'z']
    else:
        xyz = [f'x{i}' for i in range(30)]
    for i in range(data.shape[0]):
        num = alphas[i] * y[i]
        formatted_num = format_scientific(num)

        if i != 0:
            if num > 0:
                equation += f' + {formatted_num}'
            else:
                equation += f' - {format_scientific(abs(num))}'
        else:
            equation += f'{formatted_num}'

        formatted_gamma = format_scientific(gamma)

        equation += f' * e^{{-{formatted_gamma}('

        for j in range(len(x_indexes)):
            equation += f'({x[i, j]:.3f} - '
            if len(x_indexes) > 3:
                equation += f'x{j + 1})^2 '
            else:
                equation += f'{xyz[j]})^2 '

            if j != len(x_indexes) - 1:
                equation += ' + '

        equation += ')}'

    formatted_b = format_scientific(b)

    if b < 0:
        equation += f'{formatted_b}'
    else:
        equation += f' + {formatted_b}'

    equation += ' = 0'
    return equation, d



def do_Polynomial(data, x, y, matrixSize, c=1, d =2, pP=False, C=1):
    if pP:
        print(f'c: {c}\nd: {d}')

    ynk = np.zeros((matrixSize, matrixSize))
    for i in range(matrixSize):
        for j in range(matrixSize):
            ynk[i][j] = y[i] * y[j] * calc_polynomial(c, d, i, j)

    K = polynomial_kernel(matrixSize, x, c, d)

    Q = getQMatrix(matrixSize, y, K)


    solution = get_a(matrixSize, Q, y, C=C)

    # Extract the Lagrange multipliers (α)
    alphas = np.array(solution['x']).flatten()
    if pP:
        print("Lagrange Multipliers (α):", alphas)

    findMargin(ynk, alphas)
    svIndex = getSupportVectorRow(alphas)
    yk = y[svIndex]
    xk = x[svIndex]

    b = float(yk.item())

    sumK = 0
    for i in range(x.shape[0]):
        sumK+= alphas[i] * y[i] * calc_polynomial(c, d, x[i], xk)
    
    b -= sumK
    d = Polynomial(alphas, C, y, b, x, matrixSize, c, d)
    if pP:
        print(f'b: {b}')
    xyz = ['x', 'y', 'z']
    equation = ''
    for i in range(data.shape[0]):
        num = alphas[i] * y[i]
        formatted_num = format_scientific(num)
        if i != 0:
            if num > 0:
                equation += f' + {formatted_num}'
            else:
                equation += f' - {format_scientific(abs(num))}'
        else:
            equation += f'{formatted_num}'
        equation += ' * ['
        for j in range(varCount):
            if j != varCount - 1:
                if varCount <= 3:
                    equation += f'({x[i, j]:.3f}*{xyz[j]}) + '
                else:
                    equation += f'({x[i, j]:.3f}*x{j + 1}) + '
            else:
                if varCount <= 3:
                    equation += f'({x[i, j]:.3f}*{xyz[j]}'
                else:
                    equation += f'({x[i, j]:.3f}*x{j + 1})'

        equation += f') + {c}]^{d}'

    if b > 0:
        equation += f' + {format_scientific(b)}'
    else:
        equation += f' - {format_scientific(abs(b))}'
    equation += ' = 0'
    # print(type(equation))

    return equation, d


def do_Sigmoid(data, x, y, matrixSize,  a=-1.0, c = 0, pP = False, C=1):
    if a == -1.0:
        a = 1 / (100 * getAvgDot(matrixSize, x))
    if pP:
        print(f'a: {a}')
        print(f'c: {c}')
    ynk = np.zeros((matrixSize, matrixSize))
    for i in range(matrixSize):
        for j in range(matrixSize):
            ynk[i][j] = y[i] * y[j] * calc_sigmoid(x[i], x[j], a, c)

    K = sigmoid_kernel(matrixSize, x, a, c)

    eigenvalues = np.linalg.eigvals(K)
    if pP:
        print("K eigenvalues:", eigenvalues)
    if (eigenvalues < 0).any():
        return '', None
    Q = getQMatrix(matrixSize, y, K)
    print(f'Q Matrix: {Q}')
    print(f'Y: {y}')
    solution = get_a(matrixSize, Q, y, C=C)

    # Extract the Lagrange multipliers (α)
    alphas = np.array(solution['x']).flatten()
    if pP:
        print("Lagrange Multipliers (α):", alphas)

    findMargin(ynk, alphas)
    svIndex = getSupportVectorRow(alphas)
    yk = y[svIndex]
    xk = x[svIndex]

    b = float(yk.item())

    sumK = 0
    equation = ''
    for i in range(x.shape[0]):
        sumK += alphas[i] * y[i] * calc_sigmoid(x[i], xk, a, c)

    b -= sumK
    obj = Sigmoid(alphas, C, y, b, x, matrixSize, a, c)
    if pP:
        print(f'b: {b}')
    xyz = ['x', 'y', 'z']
    for i in range(data.shape[0]):
        num = alphas[i] * y[i]
        #print(num)
        formatted_num = format_scientific(num)
        if i != 0:
            if num > 0:
                equation+= f' + {num}'
            else:
                equation+= f' - {abs(num)}'
        else:
            equation+= f'{formatted_num}'
        equation+= f' * tanh({a} * '
        for j in range(varCount):
            if j != varCount - 1:
                if varCount <= 3:
                    equation += f'({x[i, j]:.3f}*{xyz[j]}) + '
                else:
                    equation += f'({x[i, j]:.3f}*x{j+1}) + '
            else:
                if varCount <= 3:
                    equation += f'({x[i, j]:.3f}*{xyz[j]})'
                else:
                    equation += f'({x[i, j]:.3f}*x{j+1})'
        equation+= f' + {c})'
    if b> 0:
        equation+= f' + {format_scientific(b)}'
    else:
        equation+= f' - {format_scientific(abs(b))}'
    equation+= ' = 0'
    return equation, obj

def do_Linear(data, x, y, matrixSize, C=1):
    K = linear_kernel(x, matrixSize)
    Q = getQMatrix(matrixSize, y, K)
    solution = get_a(matrixSize, Q, y, C)
    # Extract the Lagrange multipliers (α)
    alphas = np.array(solution['x']).flatten()
    # print(alphas)
    W = [0] * varCount
    for i in range(matrixSize):
        for j in range(varCount):
            A = float(alphas[i])
            Y = y[i]
            X = x[i,j]
            W[j] += A * Y * X
    print(f'W: {W}')
    support_vectors = [i for i, alpha in enumerate(alphas) if alpha > 1e-5]
    row = support_vectors[0]
    print('row:')
    print(row)
    sumWX = 0
    for i in range(len(W)):
        sumWX += W[i] * x[row, i]

    b = data.iloc[row, y_index] - sumWX
    obj = Linear(data.shape[0], C, W, b)
    equation = ''
    for i in range(varCount):
        if W[i] > 0:
            if i != 0:
                equation += " + "
            equation += f'{abs(W[i])}'
        else:
            if i != 0:
                equation += f' - {abs(W[i])}'
            else:
                equation += f'-{abs(W[i])}'
        equation += f'x{i + 1}'
    if b > 0:
        equation += ' + '
    equation += f'{b} = 0'
    return equation, obj

# def testPoly():
#     equations = ''
#     for i in range(1, 4):
#         # obj: RBF = do_RBF(gamma=1*(10**i), p=True)
#         for j in range(2, 4):
#             equations += f'{do_Polynomial(c=i, d=j)}\n'
#
# equations = ''
# for i in range(-3, 3):
#     obj, eq = do_RBF(1*(10**i),True)
#     equations += f'{eq}\n'


def crossValidation(data, k, kernel: str, gamma=0.1, c=0, d=2, a=0.1, C=1) -> ResultData:
    # Total number of rows in data
    n = data.shape[0]
    binSize = int(n / k)
    bins = [None] * k  # Initialize list of k bins

    # Create k bins of equal size
    for i in range(k):
        rows = list(range(i * binSize, i * binSize + binSize))
        bins[i] = data.iloc[rows].copy()  # copy() to ensure independent DataFrames
        # Debug print for each bin
        # print(f'bins[{i}]:\n{bins[i]}\n')

    totalPercentages = 0.0
    # For each fold
    for i in range(k):
        # Use the i-th bin as test data; all others as training data
        testData = bins[i]
        # print(f'TestData#{i}:\n{testData}')
        trainData = []
        for j in range(k):
            if j != i:
                trainData.append(bins[j])
        combinedTrainData = pd.concat(trainData, axis=0)

        # print(f'combinedTrainData:\n{combinedTrainData}\n')

        # Extract features and labels from training data
        X_train = getX(combinedTrainData)
        Y_train = getY(combinedTrainData)
        # Create model object based on kernel
        if kernel == 'RBF':
            # obj = getObj(combinedTrainData, kernel, X_train, Y_train, gamma=gamma)
            _, obj = do_RBF(combinedTrainData, getX(combinedTrainData), getY(combinedTrainData), combinedTrainData.shape[0], pP=False, gamma=gamma, C=C)
        elif kernel == 'Polynomial':
            # obj = getObj(combinedTrainData, kernel, X_train, Y_train, c=c, d=d)
            _, obj= do_Polynomial(combinedTrainData, getX(combinedTrainData), getY(combinedTrainData), combinedTrainData.shape[0], pP=False, c=c, d=d, C=C)
        elif kernel == 'Sigmoid':
            # obj = getObj(combinedTrainData, kernel, X_train, Y_train, c=c, a=a)
            _, obj = do_Sigmoid(combinedTrainData, getX(combinedTrainData), getY(combinedTrainData), combinedTrainData.shape[0], c=c, a=a, C=C)
        elif kernel == 'Linear':
            _, obj = do_Linear(combinedTrainData, X_train, Y_train, combinedTrainData.shape[0], C=C)
        else:
            obj = None


        # Evaluate on test set:
        true_count = 0
        false_count = 0
        confusionMatrix = {'TP':0, 'TN':0, 'FP':0, 'FN':0}
        for j in range(testData.shape[0]):
            # Get the j-th row as a DataFrame (so getX/getY work correctly)
            row_df = testData.iloc[[j]]
            xTestData = getX(row_df)
            # print(f'xTestData: {xTestData}')
            # Here, predict() and obj.calculate() must be defined appropriately.
            if obj is not None:
                prediction = predict(obj.calculate(xTestData))
                actual = getY(row_df)
                if prediction == actual:
                    true_count += 1
                    if prediction == 1:
                        confusionMatrix['TP'] += 1
                    else:
                        confusionMatrix['TN'] += 1
                else:
                    false_count += 1
                    if prediction == 1:
                        confusionMatrix['FP'] += 1
                    else:
                        confusionMatrix['FN'] += 1
        print(f'T: {true_count}, F: {false_count}')
        totalPercentages += true_count / testData.shape[0]

    # Finally, build an object on the whole data for later use:
    eq, wholeDataObj = '', None
    if kernel == 'RBF':
        # wholeDataObj = getObj(data, kernel, getX(data), getY(data), gamma=gamma, C=C)
        eq, wholeDataObj = do_RBF(data, getX(data), getY(data), data.shape[0], pP = False,  gamma=gamma, C=C)
    elif kernel == 'Polynomial':
        # wholeDataObj = getObj(data, kernel, getX(data, True), getY(data), c=c, d=d, C=C)
        eq, wholeDataObj = do_Polynomial(data, getX(data), getY(data), data.shape[0], pP= False, c= c, d = d, C=C)
    elif kernel == 'Sigmoid':
        # wholeDataObj = getObj(data, kernel, getX(data, True), getY(data), c=c, a=a, C=C)
        eq, wholeDataObj = do_Sigmoid(data, getX(data), getY(data), data.shape[0], c=c, a=a, C=C)
    elif kernel == 'Linear':
        eq, wholeDataObj = do_Linear(data, getX(data), getY(data), data.shape[0], C=C)
    else:
        wholeDataObj = None
    # return totalPercentages / k * 100.0,  wholeDataObj
    rd = ResultData(eq, wholeDataObj, confusionMatrix)
    rd.setMetrics(getMetrics(confusionMatrix))
    return rd


def getObj(data, kernel, x, y, gamma=-1.0, c=0, d=2, a=0.1, C=1):

    if kernel == 'RBF':
        obj: RBF
        _, obj = do_RBF(data, x, y, data.shape[0], gamma, False, C=C)
        # print(f'RBF Equation: {_}')
        return obj
    elif kernel == 'Polynomial':
        obj: Polynomial
        _, obj = do_Polynomial(data, x, y, data.shape[0],  c, d, False, C=C)
        # print(f'Polynomial Equation: {_}')
        return obj
    elif kernel == 'Sigmoid':
        obj:Sigmoid
        _, obj = do_Sigmoid(data, x, y, data.shape[0], a, c, False, C=C)
        print(f'Sigmoid Equation: {_}')
        if obj is not None:
            return obj
        return None
    return None


def checkAccuracy(data, kernel: str, gamma=0.1, c=0, d=2, a=0.1, C=1):
    trainData = data.head(int(4 / 5 * data.shape[0]))
    testData = data.tail(int(1 / 5 * data.shape[0]))
    eq, obj = '', None
    if kernel == 'Linear':
        eq, obj = do_Linear(trainData, getX(trainData), getY(trainData), trainData.shape[0], C=C)
    elif kernel == 'RBF':
        eq, obj = do_RBF(trainData, getX(trainData), getY(trainData), trainData.shape[0], gamma, False, C=C)
    elif kernel == 'Polynomial':
        eq, obj = do_Polynomial(trainData, getX(trainData), getY(trainData), trainData.shape[0], c, d, False, C=C)
    elif kernel == 'Sigmoid':
        eq, obj = do_Sigmoid(trainData, getX(trainData), getY(trainData), trainData.shape[0], a, c, False, C=C)

    correct = 0
    confusionMatrix = {'TP':0, 'TN':0, 'FP':0, 'FN':0}
    for i in range(testData.shape[0]):
        row_df = testData.iloc[[i]]
        xTestData = getX(row_df)
        prediction = predict(obj.calculate(xTestData))
        actual = getY(row_df)
        if prediction == actual:
            correct += 1
            if prediction == 1:
                confusionMatrix['TP'] += 1
            else:
                confusionMatrix['TN'] += 1
        else:
            if prediction == 1:
                confusionMatrix['FP'] += 1
            else:
                confusionMatrix['FN'] += 1
    percentage = correct/testData.shape[0] * 100
    rd = ResultData(eq, obj, confusionMatrix)
    rd.setMetrics(getMetrics(confusionMatrix))
    return rd

hyperparameters = {'Types Of SVM: ':['Linear', 'RBF', 'Polynomial'],
                   'C Values: ':[0.1, 1, 10, 100, 1000], #5
                   'RBF Gamma Values: ':[-1.0, 0.001, 0.01, 0.1, 1, 10, 100], #5 * 7 = 35
                   'Polynomial C Values: ':[i for i in range(1, 5)],
                   'Polynomial D Values: ': [2, 3, 4]} # 5 * 4 * 3 = 60

def tuneHyperparameter(data, k, method):
    if method not in methods:
        print('Invalid Method!')
        return

    typesOFSVM = hyperparameters['Types Of SVM: ']
    valuesC = hyperparameters['C Values: ']
    RBFgammas = hyperparameters['RBF Gamma Values: ']
    polyC = hyperparameters['Polynomial C Values: ']
    polyD = hyperparameters['Polynomial D Values: ']
    sigA = [-1.0, 0.001, 0.01, 0.1, 1]
    sigC = [i for i in range(-5, 0)]
    listOfRD:list[ResultData] = []
    for svm in typesOFSVM:
        if svm == 'RBF':
            for C in valuesC:
                for gamma in RBFgammas:
                    print(f'Type {svm}=> C: {C} || gamma: {gamma}')
                    rd:ResultData = None
                    if method == 'cv':
                        rd = crossValidation(data, k, svm, gamma=gamma, C=C)
                    elif method == 'noncv':
                        rd = checkAccuracy(data, svm, gamma=gamma, C=C)
                    #print(rd.toString(printResultType.SINGLE_LINE))
                    listOfRD.append(rd)
        elif svm == 'Polynomial':
            for C in valuesC:
                for c in polyC:
                    for d in polyD:
                        print(f'Type {svm}=> C:{C} || c:{c} || d:{d}')
                        rd:ResultData = None
                        if method == 'cv':
                            rd = crossValidation(data, k, svm, c=c, d=d, C=C)
                        elif method == 'noncv':
                            rd = checkAccuracy(data, svm, c=c, d=d, C=C)
                        #print(rd.toString(printResultType.SINGLE_LINE))
                        listOfRD.append(rd)
        elif svm == 'Sigmoid':
            for C in valuesC:
                for c in sigC:
                    for a in sigA:
                        rd:ResultData = None
                        if method == 'cv':
                            rd = crossValidation(data, k, svm, c=c, a=a, C=C)
                        elif method == 'noncv':
                            rd = checkAccuracy(data, svm, c=c, a=a, C=C)
                        #print(rd.toString(printResultType.SINGLE_LINE))
                        listOfRD.append(rd)
        elif svm == 'Linear':
            for C in valuesC:
                print(f'Type {svm}=> C:{C}')
                rd:ResultData = None
                if method == 'cv':
                    rd = crossValidation(data, k, svm, C=C)
                elif method == 'noncv':
                    rd = checkAccuracy(data, svm, C=C)
                #print(rd.toString(printResultType.SINGLE_LINE))
                listOfRD.append(rd)
    return listOfRD



def dash():
    print('='*60)

def getMetrics(confusionMatrix:dict[str:int]):
    TP = confusionMatrix['TP']
    TN = confusionMatrix['TN']
    FP = confusionMatrix['FP']
    FN = confusionMatrix['FN']
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)
    return {'ACCURACY':accuracy, 'PRECISION':precision, 'RECALL':recall, 'F1SCORE':f1_score}

rdS = []

# for i in range(1,11):
#
#
# for i in range(len(rdS)):
#     print(f'Sample Size : {rdS[i].kernel.sampleSize} =>> {rdS[i].kernel.toString()}: {rdS[i].percentage:.6f}% || Time Taken: {rdS[i].timeTaken:.6f} seconds')

print(f'Testing for Sample size of {n}')

startTime = time.time()
results = tuneHyperparameter(sampleData, 5, 'cv')#arrays of result data
endTime = time.time()

objFileName = f'{results[0].toString(printType.FILENAME)}.SO'
objFile = SVMTuningStatistics(results, hyperparameters, 'Cross Validation', int(4 / 5 * n), int(1 / 5 * n))
with open(objFileName, "wb") as file:
    pickle.dump(objFile, file)

printStats(objFile)
timeTaken = endTime - startTime
print(f"Time Taken: {timeTaken:.6f} seconds")




# with open(objFileName, "wb") as file:
    #     pickle.dump(maxObj, file)

print('Object Saved Successfully')

# print(equations)
# sampleRow = data.iloc[100, x_indexes].values
# print(obj.calculate(sampleRow))

#seperate the creating algortihm with testing. seperate the creating multiple diffrent types of kernel with its parameters with finding the best percentage

