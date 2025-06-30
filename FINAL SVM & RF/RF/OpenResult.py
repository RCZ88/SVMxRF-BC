import pickle
from TuningDataStatistics import *
def main():
    filename = '../FINAL OUTPUT/VariableAmount_10-10_TreeAmount_50-100_MaxDepth_10-15_None.txt'
    result:RFTuningStatistics = None
    with open(filename, 'rb') as file:
        result = pickle.load(file)
    forestSample = result.paramObjects[0].forests

    print(f'Hyperparameters used: {result.parameters}')
    for params in result.paramObjects:
        print(f'Parameter: {params.hyperparameter} ')
        sum = 0
        for tree in params.forests[3].trees:
            sum += tree.getMaxDepth(tree)
        print(f'Average Depth: {sum / len(params.forests[3].trees)}')
        print(f"Confusion Matrix's\nMatrix:{params.confusionMatrix}\nMetrics:{params.metrics}")
        print(f'---' * 10)




if __name__ == '__main__':
    main()