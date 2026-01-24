from ResultData import ResultData
class NLObject:
    def __init__(self, datas:list[ResultData], hyperparameters:dict, evalMethod:str, trainData:int, testData:int):
        self.datas = datas
        self.hyperparameters = hyperparameters
        self.evalMethod = evalMethod
        self.trainData = trainData
        self.testData = testData


