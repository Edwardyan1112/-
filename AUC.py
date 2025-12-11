import numpy

def AUC(y_true, y_pred):
    greater = 0
    total = 0
    
    for i in range(0, len(y_true) - 1):
        for j in range(1, len(y_true)):
            if y_true[i] != y_true[j]:
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    greater += 1
                total += 1
                
    return greater / total


def AUC_fast(y_true, y_pred):
    ranks = enumerate(sorted(zip(y_true, y_pred), key=lambda x: x[1]), start=1)
    
    positive_num = sum(y_true)
    negative_num = len(y_true) - positive_num
    positive_ranks = [x[0] for x in ranks if x[1][0] == 1]
    
    auc = (sum(positive_ranks) - positive_num * (positive_num - 1) / 2) / (positive_num * negative_num)
    return auc