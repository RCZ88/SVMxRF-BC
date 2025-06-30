from Forest import *

class ResultData:
    def __init__(self, randomForest:list[Forest], confusionMatrix, metrics):
        self.randomForest = randomForest
        self.confusionMatrix = confusionMatrix
        self.metrics = metrics