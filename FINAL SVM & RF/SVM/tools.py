from TuningDataStatistics import SVMTuningStatistics
from index import *
from svmOBJ import *


def printStats(resultFile:SVMTuningStatistics, printAll:bool = False, sortBy:metricsType = None):
    resultDatas = resultFile.datas
    hyperparameters = resultFile.hyperparameters
    if printAll:
        for r in resultDatas:
            print(r.toString(printType.SINGLE_LINE))
        print(f"Hyperparameter Tuning Sort by Metrics: {sortBy} Is Shown Above ^^^")
    print(f"HyperParameter Tuning Used: \n{hyperparameters}")
    print(f"Evaluation Method: \n{resultFile.evalMethod} with Confusion Matrix")
    print(f"Train Data: {resultFile.trainData}")
    dash()
    rbfObjects = list(filter(lambda rd : isinstance(rd.kernel, RBF), resultFile.datas))
    polyObjects = list(filter(lambda rd : isinstance(rd.kernel, Polynomial), resultFile.datas))
    linearObject = list(filter(lambda rd : isinstance(rd.kernel, Linear), resultFile.datas))
    typesOfObjects = {"OVERALL": resultDatas, "RBF":rbfObjects, "Polynomial":polyObjects, "Linear":linearObject}
    for i in resultDatas:
        print(i.kernel.shortString())
    if sortBy is None:
        rd = resultDatas
        metricses = {metricsType.ACCURACY, metricsType.PRECISION, metricsType.RECALL, metricsType.F1SCORE}
        for key, value in typesOfObjects.items():
            for metrics in metricses:
                print(f"TOP 3 {key} SVM RANKINGS (Terms Of {metrics.value}): ")
                for i in range(3):
                    print(f'{i + 1}. {rd[i].toString(printType.MEDIUM)}')
                print(f"TOP 3 {key} LOWEST SVM RANKINGS(Terms Of {metrics.value}) ")
                for i in range(3):
                    print(f'{i + 1}. {rd[len(rd) - 1 - i].toString(printType.MEDIUM)}')
                sumForAvg = sum([r.getMetrics(metrics) for r in rd])
                print(f"AVERAGE {metrics.value}: {sumForAvg/len(rd)}")
                dash()
        print(f"Data Length: {len(resultDatas)}")
    else:
        rd = sorted(resultDatas, key=lambda x: x.getMetrics(sortBy), reverse=True)
        print(f"TOP 3 SVM RANKINGS (Terms Of {sortBy.value}): ")
        for i in range(3):
            print(f'{i + 1}. {rd[i].toString(printType.MEDIUM)}')
        print(f"TOP 3 LOWEST: ")
        for i in range(3):
            print(f'{i + 1}. {rd[len(rd) - 1 - i].toString(printType.MEDIUM)}')
        sumForAvg = sum([r.getMetrics(sortBy) for r in rd])
        print(f"AVERAGE {sortBy.value}: {sumForAvg}")
        dash()

def dash():
    print('='*60)