from Forest import Forest
from ParameterOBJ import *
class RFTuningStatistics: #store all forest from a full hyperparameter tuning test
    def __init__(self, paramObjects:list[ParameterOBJ], parameters:dict, evalMethod:str, trainDataSize:int, testDataSize:int):
        self.paramObjects = paramObjects
        self.parameters = parameters
        self.evalMethod = evalMethod
        self.trainData = trainDataSize
        self.testData = testDataSize
        self.timeTaken = None

    def setTimeTaken(self, timeTaken:float):
        self.timeTaken = timeTaken

