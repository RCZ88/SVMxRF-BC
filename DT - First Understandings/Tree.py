class Node:
    def __init__(self):
        self.data = []
        self.entropy:float = -1
        self.childIG:float = -1
        self.left = None
        self.right = None
    def adData(self, data):
        self.data.append(data)
    def setLeft(self, left):
        self.left = left
        return self
    def setRight(self, right):
        self.right = right
        return self



