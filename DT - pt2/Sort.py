class Sort:
    def __init__(self, var:str, value:float, score:float, left, right, resultLeft:int, resultRight:int):
        self.var = var
        self.value = value
        self.score = score
        self.left = left
        self.right = right
        self.resultLeft = resultLeft
        self.resultRight = resultRight

    def toString(self):
        return f'{self.var} <= {self.value}'
