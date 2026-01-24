from svmOBJ import *
from index import *
class ResultData:
    def __init__(self, equation,  kernel:svmOBJ, confusionMatrix:dict[str:int]):
        self.equation = equation.strip()
        #self.percentage = percentage
        self.kernel = kernel
        self.confusionMatrix = confusionMatrix
        self.metrics = None

    def setMetrics(self, metrics:dict[str, float]):
        self.metrics = metrics


    def toString(self, type:printType):
        toString = ''
        if type == printType.FULL:
            toString = f'{self.kernel.toString()}\nConfusionMatrix:\n'
            for key, value in self.confusionMatrix.items():
                toString += f'{key}: {value}\n'
            toString += f'Metrics:\n'
            for key, value in self.metrics.items():
                toString += f'{key}: {value}\n'
            toString += f'Equation: {self.equation}\n----------------'
        elif type == printType.NO_EQUATION:
            toString = f'{self.kernel.toString()}\nConfusionMatrix:\n'
            for key, value in self.confusionMatrix.items():
                toString += f'{key}: {value}\n'
            toString += f'Metrics:\n'
            for key, value in self.metrics.items():
                toString += f'{key}: {value}\n'
        elif type == printType.SINGLE_LINE:
            toString = f'{self.kernel.toString} || CM: {self.confusionMatrix} || Metrics: {self.metrics} '
        elif type == printType.MEDIUM:
            toString = f'{self.kernel.toString()}\nConfusionMatrix:\n'
            toString += f'{self.confusionMatrix}\n'
            toString += f'Metrics:\n{self.metrics}\n'
        elif type == printType.FILENAME:
            toString += f'acc{self.getMetrics(metricsType.ACCURACY)}_pre{self.getMetrics(metricsType.PRECISION)}_rec{self.getMetrics(metricsType.RECALL)}_f1{self.getMetrics(metricsType.F1SCORE)}'
        return toString

    def getCM(self, whichMatrix):
        return self.confusionMatrix[whichMatrix.value]

    def getMetrics(self, whichMetrics:metricsType):
        return self.metrics[whichMetrics.value]


