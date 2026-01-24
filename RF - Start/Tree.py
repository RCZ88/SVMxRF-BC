from Filter import Sort
import pandas as pd
def printData(data):
    df = pd.DataFrame(data)
    arrData = df.to_numpy()
    print(arrData.flatten())

class Tree:
    def __init__(self, finalColName:str="", preSortData=None, sort:Sort=None, display=None, left:"Tree"=None, right:"Tree"=None, parent:"Tree"=None):
        self.preSortData = preSortData
        self.finalColName = finalColName
        self.sort = sort
        self.left = left
        self.display = display
        self.right = right
        self.parent = parent
        self.maxDepth = 0

    def getMaxDepth(self):
        left_depth = self.left.getMaxDepth() if self.left else 0
        right_depth = self.right.getMaxDepth() if self.right else 0
        return max(left_depth, right_depth)

    def __str__(self, level=0):
        indent = "    " * level  # Indentation for readability
        result = ""

        # If the node is a pure leaf (final classification result)
        if self.display is not None:
            result += f"{indent}[Leaf] Class: {self.display}\n"

        # If it's an internal node (not pure, needs further splitting)
        else:
            false, true = self.preSortData[self.preSortData[self.finalColName] == 0], self.preSortData[self.preSortData[self.finalColName] == 1]
            result += f"{indent}[Node] {self.sort.toString()} (Gain: {self.sort.score})\n"
            #result += f"{indent}    False Class: {false[self.finalColName].values}\n"
            #result += f"{indent}    True Class: {true[self.finalColName].values}\n"

            # Recursively print left and right branches
            if self.left is not None:
                result += f"{indent}├── Left:\n{self.left.__str__(level + 1)}"
            if self.right is not None:
                result += f"{indent}└── Right:\n{self.right.__str__(level + 1)}"

        return result

    def toString(self):
        print(self.__str__())
