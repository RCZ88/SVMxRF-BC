import numpy as np
import pandas as pd
import pickle
import time
from Sort import Sort
from Tree import Tree


filename = 'diabetes'
data = pd.read_csv(f'{filename}.csv')

finalColIndex = len(data.columns)-1
startColIndex:int = 0
sampleRowCount:int = 600

# # Convert categorical data to numerical
# countryReference = {"UK": 0, "USA": 1, "N": 2}
# data['Nationality'] = data['Nationality'].map(countryReference)
#
# goReference = {"YES": 1, "NO": 0}
# data['Go'] = data['Go'].map(goReference)

def getMax(data:list[Sort]):
    maxIG = 0
    maxData = None
    for d in data:
        if d.score > maxIG:
            maxIG = d.score
            maxData = d
    return maxData

def printData(data):
    df = pd.DataFrame(data)
    arrData = df.to_numpy()
    print(arrData.flatten())


def getSplitData(data, sortVal):
    return data[data <= sortVal], data[data > sortVal]

def sortBoolean(data, finalCol:str):
    return data[data[finalCol] == 0], data[data[finalCol] == 1]#true or false final data

def getEntropy(data, finalCol:str):
    false, true = sortBoolean(data, finalCol)
    size = float(len(data))
    pFalse = len(false)/size
    pTrue = len(true)/size
    return -pFalse * np.log2(pFalse) - pTrue * np.log2(pTrue)

def getGiniIndex(data, finalCol, p:bool = False, gName:str = "Values"): #p for print
    false, true = sortBoolean(data, finalCol)
    if p:
        print(gName)
        printData(false[finalCol])
        printData(true[finalCol])
    size = float(len(data))
    if size == 0:
        return 0
    pFalse = len(false)/size
    pTrue = len(true)/size
    GI = 1 - (pFalse ** 2 + pTrue ** 2)
    return GI

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


def findBestSort(data, fromC, untilC, finalCol:str):
    print("Parent Data: ")
    print("Gini Index: ", getGiniIndex(data, finalCol, True), "\n----------------\n")

    dataGathered = []

    for i in range(fromC, untilC+1): #column
        for j in range(len(data)): #row
            sortVal = data.iloc[j, i]
            left, right = data[data.iloc[:, i] <= sortVal], data[data.iloc[:, i] > sortVal]
            print(f"Var: {data.columns[i]} <= {sortVal} ")
            ig = getInformationGain(left, right, data, finalCol)
            dataGathered.append(Sort(data.columns[i], sortVal, ig, left, right, finalLeaveOfData(left, finalCol), finalLeaveOfData(right, finalCol)))
            print(f"Information Gained: {ig}\n")
    maxIGData = getMax(dataGathered)
    print(f'Max Information Gain: {maxIGData.toString()}')
    return maxIGData



def buildTree(data, finalColIndex:int, branchCount = 1, fromC = -1, untilC = -1, tree:Tree = None):
    if untilC == -1:
        untilC = len(data.columns)
    if fromC == -1:
        fromC = 0

    print(f"Branch Count: {branchCount}")

    if tree is None:
        tree = Tree()

    tree.preSortData = data
    finalColName = data.columns[finalColIndex]
    tree.finalColName = str(finalColName)
    afterSort = findBestSort(data, fromC, untilC, finalColName)
    tree.sort = afterSort


    if isPure(afterSort.left, finalColName):
        tree.left = Tree(preSortData=afterSort.left, sort=afterSort,
                          display=finalLeaveOfData(afterSort.left, finalColName), parent=tree,
                          finalColName=finalColName)
    else:
        tree.left = Tree(preSortData=afterSort.left, sort=afterSort, parent=tree, finalColName=finalColName)
        print(tree.toString())
        tree.left = buildTree(tree.left.preSortData, finalColIndex, branchCount + 1, fromC, untilC, tree.left)

    if isPure(afterSort.right, finalColName):
        tree.right = Tree(preSortData=afterSort.right, sort=afterSort, parent=tree,
                          display=finalLeaveOfData(afterSort.right, finalColName), finalColName=finalColName)
    else:
        tree.right = Tree(preSortData=afterSort.right, sort=afterSort, parent=tree, finalColName=finalColName)
        print(tree.toString())
        tree.right = buildTree(tree.right.preSortData, finalColIndex, branchCount + 1, fromC, untilC, tree.right)

    return tree

# findBestSort(data, startColIndex, finalColIndex)

sampleData = data.head(sampleRowCount)
startTime = time.time()
tree = buildTree(sampleData, finalColIndex, fromC = startColIndex, untilC = finalColIndex-1)
endTime = time.time()
tree.maxDepth = tree.getMaxDepth()

print(tree.toString())
print(f"Time Taken: {(endTime - startTime):.6f} seconds")

dataFileName = f"{filename}DTM{sampleRowCount}R.dt"
with open(dataFileName, "wb") as file:
    pickle.dump(tree, file)

print("Data Saved Successfully!")





