import pandas as pd
import pickle
import matplotlib.pyplot as plt
from Tree import Tree

filename = "diabetes"
data = pd.read_csv(f'{filename}.csv')

startTestRow = 201
endTestRow = 500

selectedTestData = data.sample(500)#data.iloc[startTestRow:endTestRow]

finalColumnIndex = len(selectedTestData.columns)-1

dtmFileName = 'diabetesDTM500R.dt'

with open(dtmFileName, 'rb') as file:
    DTM = pickle.load(file) #decision tree model

def useDTM(dtm:Tree, row:int):
    dataOfRow = selectedTestData.loc[row]
    count = 0
    while dtm.left is not None and dtm.right is not None:
        count+=1
        compareVar = dtm.sort.var
        compareVal = dtm.sort.value
        if dataOfRow[compareVar] <= compareVal:
            dtm = dtm.left
        else:
            dtm = dtm.right
    print(count)
    return dtm.display, count

def testAccuracy(selectedTestData, finalColumnIndex, dtm):
    tof = []
    trueCount = 0
    falseCount = 0
    columnName = selectedTestData.columns[finalColumnIndex]
    dataOfDepth:dict = {}
    print("testAccuracy")
    for row in selectedTestData.index:
        result, depth = useDTM(dtm, row)
        if depth not in dataOfDepth:
            dataOfDepth[depth] = []
        if result == selectedTestData.loc[row, columnName]:
            if result == 1:
                print("True Positive")
            else:
                print("True Negative")
            tof.append(True)
            dataOfDepth[depth].append('t')
            # print("True")
            trueCount += 1
        else:
            if result == 1:
                print("False Positive")
            else:
                print("False Negative")
            tof.append(False)
            dataOfDepth[depth].append('f')
            # print("False")
            falseCount += 1
    dataOfDepth = dict(sorted(dataOfDepth.items()))

    depthVsAccuracy = {}
    for depth in dataOfDepth:
        tc = 0
        fc = 0
        for data in dataOfDepth[depth]:
            if data == 't':
                tc += 1
            else:
                fc += 1
        accuracy = tc/len(dataOfDepth[depth]) * 100
        depthVsAccuracy[depth] = accuracy
        print(f'Depth: {depth} | Correct:{tc}, False:{fc} | Accuracy: {accuracy}%')

    generateGraph(depthVsAccuracy)
    print("SUMMARY: ")
    print("Correct Data: ", trueCount)
    print("Incorrect Data: ", falseCount)
    print("Accuracy Percentage: ", trueCount / len(selectedTestData.index) * 100, "%")

def generateGraph(depthVsAccuracy:dict):
    plt.figure(figsize=(10, 6))
    plt.plot(list(depthVsAccuracy.keys()), list(depthVsAccuracy.values()), marker='o', color='blue', label='Accuracy')

    # Labels and title
    plt.xlabel("Depth of Decision Tree")
    plt.ylabel("Accuracy (%)")
    plt.title("Decision Tree Depth vs. Accuracy")
    plt.legend()
    plt.grid(True)

    # Show the graph
    plt.show()


testAccuracy(selectedTestData, finalColumnIndex, DTM)
#print(DTM)

