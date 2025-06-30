from abc import abstractmethod

import numpy as np

class svmOBJ:
    @abstractmethod
    def __init__(self, b, C, sampleSize):
        self.b = b
        self.C = C
        self.sampleSize = sampleSize
    def toString(self):
        return None
    def shortString(self):
        return None

class NonLinear(svmOBJ):
    @abstractmethod
    def __init__(self, alphas, C, y, b, x, size):
        super().__init__(b, C, size)
        self.alphas = alphas
        self.y = y
        self.x = x



class RBF(NonLinear):
    def __init__(self, alphas, C, y, b, x, size, gamma):
        super().__init__(alphas, C, y, b, x, size)
        self.gamma = gamma

    def calculate(self, vars):
        sum = 0
        for i in range(self.sampleSize):
            sum += self.alphas[i] * self.y[i] * self.calcRBF(vars, self.x[i])
        sum+=self.b

        return sum

    def calcRBF(self, xi, xj):
        sumVar = np.sum((xi - xj)**2)
        return np.exp(-self.gamma * sumVar)

    def toString(self):
        return f'RBF: Gamma={self.gamma}, C={self.C}, SIZE={self.sampleSize}'
    def shortString(self):
        return f'RBFg{self.gamma:.3f}C{self.C}'

class Polynomial(NonLinear):
    def __init__(self, alphas, C, y, b, x, size, c, d):
        super().__init__(alphas, C, y, b, x, size)
        self.c = c
        self.d = d

    def calculate(self, vars):
        sum = 0
        for i in range(self.sampleSize):
            sum+= self.alphas[i] * self.y[i] * self.calcPoly(self.c, self.d, vars, self.x[i])
        sum+=self.b
        return sum

    def calcPoly(self, c, d, xi, xj):
        sumVar = np.sum(xi * xj)
        return (sumVar + c) ** d
    def toString(self):
        return f'Polynomial: c={self.c}, d={self.d}, C={self.C}, SIZE={self.sampleSize}'
    def shortString(self):
        return f'POLYc{self.c}d{self.d}C{self.C}'


class Sigmoid(NonLinear):
    def __init__(self, alphas, C, y, b, x, size, a, c):
        super().__init__(alphas, C, y, b, x, size)
        self.c = c
        self.a = a

    def calculate(self, vars):
        sum = 0
        for i in range(self.sampleSize):
            sum+= self.alphas[i] * self.y[i] * self.calcSigmoid(vars, self.x[i])
        return sum

    def calcSigmoid(self, xi, xj):
        sumVar = np.sum(xi * xj)
        return np.tanh(self.a * sumVar + self.c)

    def toString(self):
        return f'Sigmoid: a={self.a}, c={self.c}, C={self.C}'

    def shortString(self):
        return f'SIGa{self.a}c{self.c}C{self.C}'

class Linear(svmOBJ):
    def __init__(self, size,C, w, b):
        super().__init__(b, C, size)
        self.w = w
        self.b = b
        self.C = C
    def calculate(self, x):
        sum = 0
        print(x)
        print(self.w)
        for i in range(len(self.w)):
            sum += x[0, [i]] * self.w[i]
        sum += self.b
        return sum
    def toString(self):
        return f'Linear C = {self.C}, SIZE={self.sampleSize}'

    def shortString(self):
        return f'LINC{self.C}'

def predict(sum):
    if sum > 0:
        return 1
    elif sum < 0:
        return -1
    return None

