import numpy as np
import pandas as pd
import pickle
import time
import math

from pandas import DataFrame

from Filter import Filter
from Tree import Tree
from GENERAL import *
from Forest import Forest
from ParameterOBJ import *
from TuningDataStatistics import *
from ResultData import *
import math

#import database from csv file into pandas table
filename = 'Breast Cancer Wisconsin DB'
data = pd.read_csv(f"C:\\Users\\ASUS\\PycharmProjects\\MachineLearning\\FINAL SVM & RF\\{filename}.csv")
#id has no purpose on machine learning
data = data.drop('id', axis=1)
data = data.drop('Unnamed: 32', axis=1)
finalColIndex = 0
sampleRowCount:int = 400

data = data.sample(n=sampleRowCount)

#mapping the final col from letters to numbers
diagnosisReference = {'M': 0, 'B' : 1}
data['diagnosis'] = data['diagnosis'].map(diagnosisReference)

data = data.to_numpy()

#convert to numpy once. numpy -> 10-100x faster then pandas
"""
data.columns return all the column names from a table
data.columns[number] return the column name at the index of number
data[data.columns[number]] returns the data on that column name
"""
def scaleData(data):#standard scaler
    mean = []
    standardDeviation = []
    for columnIndex in range(data.shape[1]):
        sum = 0
        for rowIndex in range(data.shape[0]):
            sum += data[rowIndex, columnIndex]
        mean.append(sum/data.shape[0])
        sumSD = 0
        for rowIndex in range(data.shape[0]):
            sumSD += (data[rowIndex, columnIndex] - mean[columnIndex]) ** 2
        standardDeviation.append(math.sqrt(sumSD/data.shape[0]))
    for columnIndex in range(data.shape[1]):
        for rowIndex in range(data.shape[0]):
            data[rowIndex, columnIndex] = (data[rowIndex, columnIndex] - mean[columnIndex])/standardDeviation[columnIndex]

#move the final var into the last column
vars_array = np.delete(data, finalColIndex, axis=1)
scaleData(vars_array)
finalVar_array = data[:, finalColIndex].reshape(-1, 1)
#only the combined var is used. make so that the final var is on the lastIndex
combinedVar_array = np.hstack((vars_array, finalVar_array))
finalColName = combinedVar_array

def splitBoolean(data) -> tuple[np.ndarray, np.ndarray]:
    #data[:, -1] gets the data from final var column
    positives = data[data[:, -1] == 1]
    negatives = data[data[:, -1] == 0]
    return positives, negatives

def getGiniIndex(data) -> float:
    # print(f'gini index data: \n{data}')
    positive, negative = splitBoolean(data)
    pSize = positive.shape[0]
    nSize = negative.shape[0]
    totalSize = pSize + nSize
    pt = 0
    if pSize != 0:
        pt = (pSize/totalSize) ** 2
    pn = 0
    if nSize != 0:
        pn = (nSize/totalSize) ** 2
    return 1- (pt + pn)

def getInformationGain(leftData, rightData, parentData):
    if len(parentData) == 0:
        return 0
    leftGini = getGiniIndex(leftData)
    rightGini = getGiniIndex(rightData)
    ig = getGiniIndex(parentData) - (len(leftData)/len(parentData) * leftGini + len(rightData)/len(parentData) * rightGini)
    # print(f'Information Gain: {ig}')
    return ig

def isPure(data) -> bool:
    return getGiniIndex(data) == 0

def finalLeaveOfData(data):
    if isPure(data):
        return data[0, -1] #gets the final var of any row from the table(since pure, meaning all the final var of the rows are the same)
    return None

def findBestFilter(data):
    # print(data)
    varCount = data.shape[1]-1
    filters = []
    for i in range(varCount):
        selectedColumn = data[:, i]
        uniqueValues = np.unique(selectedColumn)
        for j in uniqueValues:
            leftData = data[selectedColumn <= j]
            rightData = data[selectedColumn > j]
            if len(leftData) == 0 or len(rightData) == 0:  # Skip invalid splits
                continue
            informationGain = getInformationGain(leftData, rightData, data)
            if informationGain <= 0:
                continue
            filter = Filter(i, j, informationGain, leftData, rightData, finalLeaveOfData(leftData), finalLeaveOfData(rightData))
            filters.append(filter)

    return getMaxInformationGain(filters)

def getMaxInformationGain(filters: list[Filter]) -> Filter:
    maxIG = 0
    maxFilter = None
    for filter in filters:
        if filter.score > maxIG:
            maxIG = filter.score
            maxFilter = filter
    return maxFilter

def depthLimitReached(data):
    tofCount = [0, 0]
    for row in range(data.shape[0]):
        tofCount[int(data[row, -1])] += 1

    if tofCount[0] > tofCount[1]:
        return 0
    return 1

def growTree(data, depthLimit:int = None,  currentBranchCount:int = 1, tree:Tree = None) -> Tree:
    if tree is None:
        tree = Tree()

    if len(data) == 0 or isPure(data):
        tree.display = data[0, -1] if len(data) > 0 else depthLimitReached(data)
        return tree

    filter = findBestFilter(data)
    if filter is None:  # Handle case where no valid split is found
        tree.display = depthLimitReached(data)
        return tree

    # print(filter)
    tree.filter = filter
    if isPure(filter.left):  # assuming that left data of from the filter is pure, create a branch of that sort, and stop (doesnt continue to recurse(no more branches being made))
        # creates a branch which display the final conclusion or final variable as the result from the filter
        tree.left = Tree(dataPostFilter=filter.left, filter=filter, parent=tree, display=filter.resultLeft)
    else:  # assuming tree is not yet pure, create a branch of that filter and continue creating branch for that branch
        if depthLimit is not None:
            if currentBranchCount < depthLimit:
                # creates the tree of that filter
                tree.left = Tree(dataPostFilter=filter.left, filter=filter, parent=tree)
                # continue building a tree on the left of tree.left.
                tree.left = growTree(filter.left, depthLimit, currentBranchCount + 1, tree.left)
            else:
                # creates a branch with no display, and doesn't continue creating a branch
                tree.left = Tree(dataPostFilter=filter.left, filter=filter, parent=tree,
                                 display=depthLimitReached(filter.left))
        else:
            tree.left = Tree(dataPostFilter=filter.left, filter=filter, parent=tree)
            tree.left = growTree(filter.left, depthLimit, currentBranchCount + 1, tree.left)

    if isPure(filter.right):  # assuming that tree
        tree.right = Tree(dataPostFilter=filter.right, filter=filter, parent=tree, display=filter.resultRight)
    else:
        if depthLimit is not None:
            if currentBranchCount < depthLimit:
                tree.right = Tree(dataPostFilter=filter.right, filter=filter, parent=tree)
                tree.right = growTree(filter.right, depthLimit, currentBranchCount + 1, tree.right)
            else:
                tree.right = Tree(dataPostFilter=filter.right, filter=filter, parent=tree,
                                  display=depthLimitReached(filter.right))
        else:
            tree.right = Tree(dataPostFilter=filter.right, filter=filter, parent=tree)
            tree.right = growTree(filter.right, depthLimit, currentBranchCount + 1, tree.right)
        # print(tree.toString())
    return tree


def growForest(data, depthLimit:int = None, varPerBootstrap:int = -1, treeAmount:int = 5):
    if varPerBootstrap == -1:
        #gets square root of the amount of variables.
        varPerBootstrap = math.floor((data.shape[1]-1) ** (1/2))

    variables = data[:, :-1]
    finalVar = data[:, -1].reshape(-1, 1)
    trees = []
    for i in range(treeAmount):
        #np.random.choice return an array with randomly selected indexes.
        randomRowIndexes = np.random.choice(variables.shape[0], size = variables.shape[0], replace=True)
        randomColumnIndexes = np.random.choice(variables.shape[1], size=varPerBootstrap, replace=False)
        bootstrapData = variables[randomRowIndexes, :][:, randomColumnIndexes]
        bootstrapData = np.hstack((bootstrapData, finalVar[randomRowIndexes, :]))
        # print(bootstrapData, f"\n{bootstrapData.shape}\n")
        tree = growTree(bootstrapData, depthLimit)
        trees.append(tree)
    sumMD = 0
    for tree in trees:
        sumMD+= tree.getMaxDepth(tree)
    print(f'Average: {sumMD/len(trees)}')
    forest = Forest(trees, depthLimit, varPerBootstrap)
    return forest

def useTree(testData, tree):
    if tree.left is None and tree.right is None:
        return tree.display
    else:
        if testData[tree.filter.varIndex] <= tree.filter.value:
            return useTree(testData, tree.left)
        else:
            return useTree(testData, tree.right)

def useForest(testData, forest:Forest):
    PoN = [0, 0]
    for tree in forest.trees:
        PoN[int(useTree(testData, tree))] += 1
    # print(f'Tree Voting: Positive={PoN[0]} votes, Negative={PoN[1]} votes')
    if PoN[0] > PoN[1]:
        return 0
    else:
        return 1

def crossValidation(data, kFold:int = 5, depthLimit:int = None, varPerBootstrap:int = -1, treeAmount:int = 5):
    #np.arange retrieves the row index of the data
    indices = np.arange(data.shape[0])
    #seed ensure that the shuffle remains the same for each usage off the cross validation method
    np.random.seed(42)
    #shuffling the index from the variable indicies
    np.random.shuffle(indices)
    #arrange the rows according to the shuffle
    data = data[indices, :]
    trainingData = []
    testData = []
    chunkSize = data.shape[0] // kFold
    for i in range(kFold):
        trainDataSeperated = []
        for j in range(kFold):
            if j == i:
                testData.append(data[j*chunkSize:(j+1)*chunkSize])
            else:
                #print(f'{j * chunkSize} - {(j + 1) * chunkSize}')
                trainDataSeperated.append(data[j*chunkSize:(j+1)*chunkSize])
        trainingData.append(np.vstack(trainDataSeperated))
        # print(np.vstack(trainDataSeperated).shape[0])
    confusionMatrix = getConfusionMatrix()
    forests = []
    for i in range(kFold):
        #print(f'\nTraining data K-{i+1} \n{trainingData[i]}\n size: {trainingData[i].shape[0]} * {trainingData[1].shape[1]}')
        start = time.time()
        forest = growForest(trainingData[i], depthLimit, varPerBootstrap, treeAmount)
        end = time.time()
        timeTaken = end - start
        print(f'Time Taken (to Build Forest): {timeTaken} Seconds')
        td = testData[i]
        start = time.time()
        for j in range(td.shape[0]):
            result = useForest(td[j, :], forest)
            updateMatrix(result, td[j, -1], confusionMatrix)
        end = time.time()
        timeTaken = end - start
        print(f'Time Taken(Use Forest): {timeTaken} Seconds')
        forests.append(forest)
    return calculateMetrics(confusionMatrix), forests, confusionMatrix

def updateMatrix(output, expected, confusionMatrix):
    if output == 1 and expected == 1:
        confusionMatrix['TP'] += 1
    elif output == 0 and expected == 0:
        confusionMatrix['TN'] += 1
    elif output == 1 and expected == 0:
        confusionMatrix['FP'] += 1
    elif output == 0 and expected == 1:
        confusionMatrix['FN'] += 1

def getConfusionMatrix():
    return {'TP':0, 'TN':0, 'FP':0, 'FN':0}

def calculateMetrics(confusionMatrix):
    TP = confusionMatrix['TP']
    TN = confusionMatrix['TN']
    FP = confusionMatrix['FP']
    FN = confusionMatrix['FN']
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)
    return {'ACCURACY': accuracy, 'PRECISION': precision, 'RECALL': recall, 'F1SCORE': f1_score}

hyperparameters = {"Variable Amount":[-1, 10, 15], "Tree Amount":[50, 100], "Max Depth":[10, 15, None]}

def hyperParameterTuning(data, hyperparameters, kFold) -> RFTuningStatistics:
    #each index for one hyperparameter
    pos = []
    for va in hyperparameters["Variable Amount"]:
        for ta in hyperparameters["Tree Amount"]:
            for md in hyperparameters["Max Depth"]:
                print(f"Evaluating Random Forest with the following hyperparameters:\n"
                      f"Variable(Column) Per Bootstrap: {va}\n"
                      f"Tree Amount: {ta}\n"
                      f"Max Depth: {md}")
                metricsFound, forests, confusionMatrix = crossValidation(data, kFold, md, va, ta)
                print(metricsFound, "\n\n")
                pos.append(ParameterOBJ(forests, {"Variable Amount":va, "Tree Amount":ta, "Max Depth":md}, metricsFound, confusionMatrix))
    tds = RFTuningStatistics(pos, hyperparameters, 'Confusion Matrix & Cross Validation', data.shape[0] * 4/kFold, data.shape[0]/kFold)
    return tds

#below are methods to generate the filename of the object

def summarize_value_list(values):
    if all(isinstance(v, (int, float)) for v in values if v is not None):
        numeric_vals = [v for v in values if v is not None]
        min_val = min(numeric_vals)
        max_val = max(numeric_vals)
        return f"{min_val}-{max_val}" + ("_None" if None in values else "")
    else:
        return "_".join(str(v) if v is not None else "None" for v in values)

def create_summary_filename(hyperparams):
    parts = []
    for key, values in hyperparams.items():
        key_clean = key.replace(" ", "").replace(":", "")
        val_summary = summarize_value_list(values)
        parts.append(f"{key_clean}_{val_summary}")
    filename = "_".join(parts) + ".RFO"
    return filename

"""
CHECKS BEFORE RUNNING: 
1. Check if the filepath is correct
2. Select the data size to use (make sure divisible by kFold)
3. Map final variable if necessary 
4. Final Var Index is correct
"""
timeStart = time.time()
tds:RFTuningStatistics = hyperParameterTuning(combinedVar_array, hyperparameters, 5)
timeEnd = time.time()
timeTaken = timeEnd - timeStart
tds.setTimeTaken(timeTaken)
print(f'Time Taken: {timeTaken}')

dataFileName = create_summary_filename(hyperparameters)
with open(dataFileName, "wb") as file:
    pickle.dump(tds, file)

