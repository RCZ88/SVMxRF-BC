from enum import Enum

class cmType(Enum):
    TF = 'TF'
    TP = 'TP'
    FP = 'FP'
    FN = 'FN'

class metricsType(Enum):
    ACCURACY = 'ACCURACY'
    PRECISION = 'PRECISION'
    RECALL = 'RECALL'
    F1SCORE = 'F1SCORE'


#print for result data
class printType(Enum):
    FULL = 'FULL'
    NO_EQUATION = 'NO_EQUATION'
    SINGLE_LINE = 'SINGLE_LINE'
    MEDIUM = 'MEDIUM'
    FILENAME = 'FILENAME'



