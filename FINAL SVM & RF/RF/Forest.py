from Tree import Tree
from typing import List
class Forest:
    def __init__(self, trees:List[Tree], maxDepth, varPerBootstrap):
        self.trees = trees
        self.bootstrapCount = len(trees)
        self.maxDepth = maxDepth
        self.varPerBootstrap = varPerBootstrap

