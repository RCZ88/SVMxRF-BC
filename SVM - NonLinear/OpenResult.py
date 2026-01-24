import pickle
from index import *
import NLObject
from tools import *

def main():
    filename = 'acc1.0_pre1.0_rec1.0_f11.0.SO'
    result = None
    with open(filename, 'rb') as file:
        result = pickle.load(file)
    print(printStats(result))


if __name__ == '__main__':
    main()