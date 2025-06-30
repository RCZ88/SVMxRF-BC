from Filter import Filter
import pandas as pd
def printData(data):
    df = pd.DataFrame(data)
    arrData = df.to_numpy()
    print(arrData.flatten())

class Tree:
    def __init__(self, finalColName:str="", dataPostFilter=None, filter:Filter=None, display=None, left: "Tree"=None, right: "Tree"=None, parent: "Tree"=None):
        self.dataPostFilter = dataPostFilter
        self.finalColName = finalColName
        self.filter = filter #used if node is not pure. Displays the comparison (ex. x<2)
        self.left = left
        self.display = display #used if node is pure(leaf). displays the solution.
        self.right = right
        self.parent = parent
        self.maxDepth = 0

    def getMaxDepth(self, tree):
        if tree is None:
            return 0
        if tree.left is None and tree.right is None:
            return 1
        return 1 + max(self.getMaxDepth(tree.left), self.getMaxDepth(tree.right))

    def __str__(self, level=0):
        indent = "    " * level
        result = f"{indent}Tree: \n"

        # Add null checks to prevent NoneType errors
        if self.filter:
            result += f"{indent}  Sort: {self.filter.varIndex} <= {self.filter.value}\n"

        if self.display is not None:
            result += f"{indent}  Display: {self.display}\n"

        if self.filter.score is not None:
            result += f"{indent}  Score: {self.filter.score}\n"

        if self.dataPostFilter is not None:
            # Safe access with try-except to handle potential issues
            try:
                false_data = self.dataPostFilter[self.dataPostFilter[:, -1] == 0]
                true_data = self.dataPostFilter[self.dataPostFilter[:, -1] == 1]
                result += f"{indent}  Data Size: {len(self.dataPostFilter)}\n"
                result += f"{indent}  False Count: {len(false_data)}\n"
                result += f"{indent}  True Count: {len(true_data)}\n"
            except Exception as e:
                result += f"{indent}  Error accessing data: {str(e)}\n"

        if self.left:
            result += f"{indent}├── Left:\n{self.left.__str__(level + 1)}"

        if self.right:
            result += f"{indent}└── Right:\n{self.right.__str__(level + 1)}"

        return result

    def toString(self):
        print(self.__str__())
