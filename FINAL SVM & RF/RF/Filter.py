class Filter:
    def __init__(self, varIndex: int, value:float, score:float, left, right, resultLeft:int, resultRight:int):
        self.varIndex = varIndex
        self.value = value #which row of the column. value of variable in data
        self.score = score  #information gain
        self.left = left #left data
        self.right = right #right data
        self.resultLeft = resultLeft #true or false
        self.resultRight = resultRight #true or false

    def __str__(self):
        return (f"Filter(varIndex={self.varIndex}, value={self.value}, score={self.score:.4f}, "
                f"resultLeft={self.resultLeft}, resultRight={self.resultRight})")

