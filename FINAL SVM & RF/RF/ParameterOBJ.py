from Forest import Forest
class ParameterOBJ:
    def __init__(self, forests:list[Forest], hyperparameter:dict, metrics:dict, confusionMatrix:dict):
        self.forests = forests
        self.hyperparameter = hyperparameter
        self.metrics = metrics
        self.confusionMatrix = confusionMatrix
