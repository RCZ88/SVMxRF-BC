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
from TuningDataStatistics import *
from ResultData import *
import math


filename = 'Breast Cancer Wisconsin DB'
data = pd.read_csv(f"C:\\Users\\ASUS\\PycharmProjects\\MachineLearning\\FINAL SVM & RF\\{filename}.csv")
data.drop('id', axis=1, inplace=True)
print(data)


finalColIndex = 0
startColIndex:int = 2
sampleRowCount:int = 300
numberOfBootstraps:int = 20
varPerBootstrap:int = 3

# # Convert categorical data to numerical
# countryReference = {"UK": 0, "USA": 1, "N": 2}
# data['Nationality'] = data['Nationality'].map(countryReference)
#
goReference = {"YES": 1, "NO": 0}
diagnosisReference = {'M': 0, 'B' : 1}
data['diagnosis'] = data['diagnosis'].map(diagnosisReference)


def getMax(data:list[Filter]):
    maxIG = 0
    maxData = None
    for d in data:
        if d.score > maxIG:
            maxIG = d.score
            maxData = d
    if maxData != None:
        return maxData, True
    else:
        print("Unable to Gain any information from the data!")
        return data[0], False



def getSplitData(data, sortVal):
    return data[data <= sortVal], data[data > sortVal]


def sortBoolean(data, finalCol: str):
    if data is None or data.empty:
        # Return empty DataFrames if input is empty
        empty_df = pd.DataFrame()
        return empty_df, empty_df
    return data[data[finalCol] == 0], data[data[finalCol] == 1]  # true or false final data

def getEntropy(data, finalCol:str):
    false, true = sortBoolean(data, finalCol)
    size = float(len(data))
    pFalse = len(false)/size
    pTrue = len(true)/size
    return -pFalse * np.log2(pFalse) - pTrue * np.log2(pTrue)

def getGiniIndex(data, finalCol:str, p:bool = False, gName:str = "Values"): #p for print
    print(f'Data Size: {data.shape[0]}')
    print(f'Data on GINI INDEX METHOD\n{data}')
    size = float(len(data))
    if size == 0:
        return 0
    false, true = sortBoolean(data, finalCol)
    if p:
        if data is not None and data.shape[0] > 0:
            print(gName)
            printData(false[finalCol])
            printData(true[finalCol])
        else:
            print("No Data fits the sort")

    pFalse = len(false)/size
    pTrue = len(true)/size
    GI = 1 - (pFalse ** 2 + pTrue ** 2)
    return GI

def printData(data):
    df = pd.DataFrame(data)
    arrData = df.to_numpy()
    print(arrData.flatten())

def getInformationGain(left, right, parent, finalCol:str):
    leftGini = getGiniIndex(left, finalCol, True, "Left Data:")
    rightGini = getGiniIndex(right, finalCol, True, "Right Data:")
    print(f"lG: {leftGini}, rG: {rightGini}")
    return getGiniIndex(parent, finalCol) - len(left)/len(parent) * leftGini - len(right)/len(parent) * rightGini

def isPure(data, finalCol):
    return getGiniIndex(data, finalCol) == 0.0

def finalLeaveOfData(data, finalCol:str):
    if data.empty:
        return None  # No data case
    if isPure(data, finalCol):  # Check if all values are the same
        return data[finalCol].iloc[0]  # Return that unique value
    return None  # Mixed values (error case)


def findBestFilter(data, finalCol:str):
    print("FINDING THE BEST SORT")
    print("Parent Data: ")
    print("Gini Index: ", getGiniIndex(data, finalCol, True), "\n----------------\n")
    print(data)
    dataGathered = []
    data_array = data.drop(columns=[finalCol])
    data
    for i in range(len(data_array)): #column
        for j in range(len(data)):  # row
            sortVal = data.iloc[j, i]
            left, right = data[data.iloc[:, i] <= sortVal], data[data.iloc[:, i] > sortVal]
            print(f"Var: {data.columns[i]} <= {sortVal} ")
            ig = getInformationGain(left, right, data, finalCol)
            dataGathered.append(Filter(data.columns[i], sortVal, ig, left, right, finalLeaveOfData(left, finalCol),
                                       finalLeaveOfData(right, finalCol)))
            print(f"Information Gained: {ig}\n")
    maxIGData, found = getMax(dataGathered)
    if found:
        print(f'Max Information Gain: {maxIGData.toString()}')
    return maxIGData, found

def depthLimitReached(data, finalCol:str):
    trueCount = 0
    falseCount = 0
    for i in range(data.shape[0]-1): #row amount
        if data.iloc[i][finalCol] == 1:
            trueCount += 1
        else:
            falseCount += 1
    if trueCount > falseCount:
        return 1
    return 0

def buildTree(data, finalColName:str, maxDepth, branchCount = 1, tree:Tree = None):


    print(f"Branch Count: {branchCount}")

    if tree is None:
        tree = Tree()

    tree.finalColName = str(finalColName)
    afterSort, found = findBestFilter(data, finalColName)
    if not found:
        return tree
    tree.filter = afterSort


    if isPure(afterSort.left, finalColName):
        tree.left = Tree(dataPostFilter=afterSort.left, filter=afterSort,
                         display=finalLeaveOfData(afterSort.left, finalColName), parent=tree,
                         finalColName=finalColName)
    else:
        if maxDepth is not None:
            if branchCount < maxDepth:
                tree.left = Tree(dataPostFilter=afterSort.left, filter=afterSort, parent=tree, finalColName=finalColName)
                print(tree.toString())
                tree.left = buildTree(tree.left.dataPostFilter, finalColName, maxDepth, branchCount + 1, tree.left)
            else:
                tree.left = Tree(dataPostFilter=afterSort.left, filter=afterSort,
                                 display=depthLimitReached(afterSort.left, finalColName), parent=tree,
                                 finalColName=finalColName)
        else:
            tree.left = Tree(dataPostFilter=afterSort.left, filter=afterSort, parent=tree, finalColName=finalColName)
            print(tree.toString())
            tree.left = buildTree(tree.left.dataPostFilter, finalColName, maxDepth, branchCount + 1, tree.left)


    if isPure(afterSort.right, finalColName):
        tree.right = Tree(dataPostFilter=afterSort.right, filter=afterSort, parent=tree,
                          display=finalLeaveOfData(afterSort.right, finalColName), finalColName=finalColName)
    else:
        if maxDepth is not None:
            if branchCount < maxDepth:
                tree.right = Tree(dataPostFilter=afterSort.right, filter=afterSort, parent=tree, finalColName=finalColName)
                print(tree.toString())
                tree.right = buildTree(tree.right.dataPostFilter, finalColName, maxDepth, branchCount + 1, tree.right)
            else:
                tree.right = Tree()
        else:
            tree.right = Tree(dataPostFilter=afterSort.right, filter=afterSort, parent=tree, finalColName=finalColName)
            print(tree.toString())
            tree.right = buildTree(tree.right.dataPostFilter, finalColName, maxDepth, branchCount + 1, tree.right)

    return tree

# findBestSort(data, startColIndex, finalColIndex)

sampleData = data.head(sampleRowCount)
excludeFinalCol = sampleData.drop(columns = data.columns[finalColIndex])
finalColName = sampleData.columns[finalColIndex]
finalColData = sampleData[finalColName]

def createForest(numberOfBootstraps, maxDepth, varPerBootstrap):

    treeBootstraps = []
    startTime = time.time()
    for i in range(numberOfBootstraps):
        rand_columns = excludeFinalCol.sample(n=varPerBootstrap, axis=1)  # get the random columns
        rand_columns.insert(0, finalColName, finalColData)
        print(f'#{i}-Random Tree: ')
        for col in rand_columns.columns:
            print(f"{col} ", end="")
        tree = buildTree(rand_columns, finalColName, maxDepth)
        treeBootstraps.append(tree)
    endTime = time.time()
    randomForest = Forest(treeBootstraps, maxDepth)
    print(f"Time Taken: {(endTime - startTime):.6f} seconds")
    return randomForest

def useDTM(selectedTestData, dtm:Tree, row:int):
    dataOfRow = selectedTestData.iloc[row]
    #print(dataOfRow)
    count = 0
    while dtm.left is not None and dtm.right is not None:
        count+=1
        compareVar = dtm.filter.var
        compareVal = dtm.filter.value
        #print(compareVar, compareVal)
        if dataOfRow[compareVar] <= compareVal:
            dtm = dtm.left
        else:
            dtm = dtm.right
    #print(count)
    return dtm.display, count

def exploreForest(selectedTestData, forest, row):
    results = []
    true = 0
    false = 0
    for tree in forest.trees:
        output, depth = useDTM(selectedTestData, tree, row)
        if output == 1:
            true += 1
            results.append(1)
        else:
            false += 1
            results.append(0)
    mvp = max(true/(true+false), false/(true+false))
    print(f'Major Voting Percentage: {mvp}')

    if true > false:
        return 1
    return 0

def crossValidation(k, selectedTestData:DataFrame, numOfBootstraps, maxDepth, varPerBootstrap):
    trainData = []
    testData = []
    sectionSize = selectedTestData.shape[0]/k
    print(f'Section Size: {sectionSize}')
    for section in range(k):
        trainDataSeperated = []
        for chunk in range(k):
            start = int(chunk * sectionSize)
            end = int(start + sectionSize)
            if chunk != section:
                trainDataSeperated.append(selectedTestData.iloc[start:end])
            else:
                testData.append(selectedTestData.iloc[start:end])
        trainData.append(pd.concat(trainDataSeperated))
        print(f'TrainData{section} (Size: {trainData[section].shape[0]}): \n{trainData}')
    confusionMatrix = getEmptyCM()
    allForest = []
    for i in range(k):
        print(f"CREATING FOREST FOR: trainData{i}")
        forest:Forest = createForest(numOfBootstraps, maxDepth, varPerBootstrap)
        allForest.append(forest)
        for j in range(testData[i].shape[0]):
            result = exploreForest(testData[i], forest, j)
            confusionMatrix[getConfusionCategory(result, data.iloc[j, finalColIndex])] += 1
    metrics = getMetrics(confusionMatrix)
    print(f'metrics: {metrics}')
    generalStoreForest = ResultData(allForest, confusionMatrix, metrics)
    return generalStoreForest

def getConfusionCategory(prediction, actual):
    if prediction == actual:
        if prediction == 1:
            return 'TP'
        else:
            return 'TN'
    else:
        if prediction == 1:
            return 'FP'
        else:
            return 'FN'



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

varCount = excludeFinalCol.shape[1]
varPerTree = math.floor(math.sqrt(varCount))
hyperparameters = {"Variable Amount: ": [varPerTree], "Boostrap Amount: ": [15 * i for i in range(1,5)], "Max Depth: ": [None, 5, 10, 20, 30, 50]}
h = {"Variable Amount: ": [varPerTree], "Boostrap Amount: ": [5,10, 15, 20], "Max Depth: ": [10, None]}

def hyperparameterTuning(data, hyperparameters:dict[str:int], k):
    variableAmount = hyperparameters["Variable Amount: "]
    bootstrapAmount = hyperparameters["Boostrap Amount: "]
    maxDepth = hyperparameters["Max Depth: "]

    forests = []
    for va in variableAmount:
        for ba in bootstrapAmount:
            for md in maxDepth:
                forests.append(crossValidation(k, data, ba, md, va))

    finalDataGathered = RFTuningStatistics(forests, hyperparameters, 'CV', data.shape[0] * 4/5, data.shape[0]/5)
    return finalDataGathered

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
    filename = "_".join(parts) + ".txt"
    return filename

startTime = time.time()
fileObj = hyperparameterTuning(data.sample(n=450), h, 5)
endTime = time.time()

timeTaken = endTime - startTime

dataFileName = create_summary_filename(h)
with open(dataFileName, "wb") as file:
    pickle.dump(fileObj, file)

print("Data Saved Successfully!")



