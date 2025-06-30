from ResultData import ResultData
class SVMTuningStatistics:
    def __init__(self, datas:list[ResultData], hyperparameters:dict, evalMethod:str, trainData:int, testData:int):
        self.datas = datas
        self.hyperparameters = hyperparameters #hyperparameter used
        self.evalMethod = evalMethod #either cross validation or non cv
        self.trainData = trainData #amt of data used to train(row Count)
        self.testData = testData #amt of data used to test(row Count)


