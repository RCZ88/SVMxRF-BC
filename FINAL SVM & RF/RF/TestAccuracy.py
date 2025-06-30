import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pandas import DataFrame

from Tree import Tree
from Forest import Forest
from CreateForest import createForest
from GENERAL import *


filename = "Breast Cancer Wisconsin DB"
data = pd.read_csv(f'{filename}.csv')

startTestRow = 201
endTestRow = 500

selectedTestData = data.sample(500)#data.iloc[startTestRow:endTestRow]

diagnosisReference = {'M': 0, 'B' : 1}
selectedTestData['diagnosis'] = selectedTestData['diagnosis'].map(diagnosisReference)

finalColumnIndex = 1

dtmFileName = 'Breast Cancer Wisconsin DBRFData50BS.rf'

with open(dtmFileName, 'rb') as file:
    RForest:Forest = pickle.load(file) #decision tree model

forest = RForest.trees


def useDTM(dtm:Tree, row:int):
    dataOfRow = selectedTestData.loc[row]
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


# def testAccuracy(selectedTestData, finalColumnIndex, dtm):
#     tof = []
#     trueCount = 0
#     falseCount = 0
#     columnName = selectedTestData.columns[finalColumnIndex]
#     dataOfDepth:dict = {}
#     print("testAccuracy")
#     for row in selectedTestData.index:
#         result, depth = useDTM(dtm, row)
#         if depth not in dataOfDepth:
#             dataOfDepth[depth] = []
#         if result == selectedTestData.loc[row, columnName]:
#             tof.append(True)
#             dataOfDepth[depth].append('t')
#             # print("True")
#             trueCount += 1
#         else:
#             tof.append(False)
#             dataOfDepth[depth].append('f')
#             # print("False")
#             falseCount += 1
#     dataOfDepth = dict(sorted(dataOfDepth.items()))
#
#     depthVsAccuracy = {}
#     for depth in dataOfDepth:
#         tc = 0
#         fc = 0
#         for data in dataOfDepth[depth]:
#             if data == 't':
#                 tc += 1
#             else:
#                 fc += 1
#         accuracy = tc/len(dataOfDepth[depth]) * 100
#         depthVsAccuracy[depth] = accuracy
#         print(f'Depth: {depth} | Correct:{tc}, False:{fc} | Accuracy: {accuracy}%')
#
#     generateGraph(depthVsAccuracy)
#     print("SUMMARY: ")
#     print("Correct Data: ", trueCount)
#     print("Incorrect Data: ", falseCount)
#     print("Accuracy Percentage: ", trueCount / len(selectedTestData.index) * 100, "%")


def exploreForest(forest, row, result:[str]):
    results = []
    true = 0
    false = 0
    for tree in forest:
        output, depth = useDTM(tree, row)
        if output == 1:
            true += 1
            results.append(1)
        else:
            false += 1
            results.append(0)
    print(f"Forest Results: {results}\n{result[1]}:{result[0]} Ratio = {true}:{false}")
    mvp = max(true/(true+false), false/(true+false))
    print(f'Major Voting Percentage: {mvp}')

    if true > false:
        return 1
    return 0


def testAccuracy(forest, selectedTestData, finalVars: [str]):
    columnName = selectedTestData.columns[finalColumnIndex]
    print(columnName)
    tc = 0
    fc = 0

    confusionMatrix = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    for row in selectedTestData.index:
        print(f'Sample Row: #{row}\n---------------')
        result = exploreForest(forest, row, finalVars)
        if result == selectedTestData.loc[row, columnName]:
            if result == 1:
                print("True Positive")
                confusionMatrix['TP'] += 1
            else:
                print("True Negative")
                confusionMatrix['TN'] += 1
            print("Correct")
            tc += 1
        else:
            if result == 1:
                print("False Positive")
                confusionMatrix['FP'] += 1
            else:
                print("False Negative")
                confusionMatrix['FN'] += 1
            print("Incorrect")
            fc += 1
        print("\n\n")
    getMetrics(confusionMatrix)
    print(f"Correct: {tc}, Incorrect: {fc}")
    print(f"Percentage = {tc / len(selectedTestData.index) * 100}%")

def crossValiation(k, selectedTestData:DataFrame, numOfBootstraps, finalVars:[str]):
    trainData = []
    testData = []
    sectionSize = selectedTestData.shape[0]/k
    for section in range(k):
        trainDataSeperated = []
        for chunk in range(k):
            start = chunk * sectionSize
            end = start + sectionSize
            if chunk != section:
                trainDataSeperated.append(selectedTestData.iloc[start:end])
            else:
                testData.append(selectedTestData.iloc[start:end])
        trainData.append(pd.concat(trainDataSeperated))
    confusionMatrix = getEmptyCM()
    for i in range(k):
        forest = createForest(numOfBootstraps)
        result = exploreForest(forest, testData[i], finalVars)
        confusionMatrix[getConfusionCategory(result, )]




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


def generateGraph(table:dict, title, xLabel, yLabel):
    plt.figure(figsize=(10, 6))
    plt.plot(list(table.keys()), list(table.values()), marker='o', color='blue', label='Accuracy')

    # Labels and title
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Show the graph
    plt.show()

finalVars = ["Malignant", "Benign"]
testAccuracy(forest, selectedTestData, finalVars)
# testAccuracy(selectedTestData, finalColumnIndex, DTM)
# #print(DTM)

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
