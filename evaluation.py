# evaluation.py
# Contains the evaluation methods in one file for KNN, K-Means and Naive Bayes
# The function works for any classes
# 0 = not dynasty, 1 = dynasty
# Also works for (none, thin, fat, obese)

# Metrics:
# Precision
# Balanced Accuracy
# Matthews correlation coefficient

import math
 
# names used when printing the 2 class version
BINARY_NAMES = {0: "not dynasty", 1: "dynasty"}

# confusion matrix
 
def getClasses(yTrue, yPred):
    # all the different labels that appear, sorted so the order is always the same
    return sorted(set(yTrue) | set(yPred))
 
 
def confusionMatrix(yTrue, yPred, classes):
    # matrix[i][j] = how many rows were really classes[i] but we predicted classes[j]
    # so the correct answers are on the diagonal (i == j)
    position = {}
    for index, c in enumerate(classes):
        position[c] = index
 
    size = len(classes)
    matrix = []
    for i in range(size):
        matrix.append([0] * size)
 
    for t, p in zip(yTrue, yPred):
        matrix[position[t]][position[p]] += 1
 
    return matrix
 
 
def printConfusionMatrix(matrix, classes, classNames):
    print("  Confusion matrix (rows = real, columns = predicted)")
    header = "  " + " " * 14
    for c in classes:
        header += f"{classNames.get(c, str(c)):>14}"
    print(header)
 
    for i, c in enumerate(classes):
        line = f"  {classNames.get(c, str(c)):<14}"
        for j in range(len(classes)):
            line += f"{matrix[i][j]:>14,}"
        print(line)
 
 
# helper counts from the matrix
 
def rowTotal(matrix, i):
    # how many rows REALLY belong to class i
    return sum(matrix[i])
 
 
def columnTotal(matrix, j):
    # how many rows we PREDICTED as class j
    total = 0
    for i in range(len(matrix)):
        total += matrix[i][j]
    return total
 
 
def matrixTotal(matrix):
    total = 0
    for i in range(len(matrix)):
        total += sum(matrix[i])
    return total
 
 
def average(values):
    return sum(values) / len(values) if len(values) > 0 else 0
 
 
# accuracy
 
def accuracy(matrix):
    # correct answers are the diagonal, divided by all the rows
    correct = 0
    for i in range(len(matrix)):
        correct += matrix[i][i]
    return correct / matrixTotal(matrix)
 
 
# Precision
 
def precisionPerClass(matrix):
    # precision of class k = TP_k / (TP_k + FP_k)
    # = (rows correctly called k) / (all rows we called k)
    values = []
    for k in range(len(matrix)):
        predicted = columnTotal(matrix, k)
        # if we never predicted this class we use 0 so we dont divide by zero
        values.append(matrix[k][k] / predicted if predicted > 0 else 0)
    return values
 
 
def macroPrecision(matrix):
    # plain average of the precision of every class
    # rare classes count the same as the big ones
    return average(precisionPerClass(matrix))
 
 
# Balanced accuracy
 
def recallPerClass(matrix):
    # recall of class k = TP_k / (TP_k + FN_k)
    # = (rows correctly called k) / (all rows that really are k)
    values = []
    for k in range(len(matrix)):
        real = rowTotal(matrix, k)
        values.append(matrix[k][k] / real if real > 0 else 0)
    return values
 
 
def balancedAccuracy(matrix):
    # average of the recall of every class
    # always guessing the biggest class gets only 1 / (number of classes)
    return average(recallPerClass(matrix))
 
 
# MCC
 
def mcc(matrix):
    # multi class MCC formula:
    #   (c * s - sum(p_k * t_k)) / sqrt((s^2 - sum(p_k^2)) * (s^2 - sum(t_k^2)))
    # c = correct rows, s = all rows,
    # p_k = rows we predicted as class k, t_k = rows that really are class k
    s = matrixTotal(matrix)
    c = 0
    for i in range(len(matrix)):
        c += matrix[i][i]
 
    sumPT = 0
    sumPP = 0
    sumTT = 0
    for k in range(len(matrix)):
        p = columnTotal(matrix, k)
        t = rowTotal(matrix, k)
        sumPT += p * t
        sumPP += p * p
        sumTT += t * t
 
    top = c * s - sumPT
    bottom = math.sqrt((s * s - sumPP) * (s * s - sumTT))
 
    # bottom is 0 when we predict only one class, then MCC is just 0
    if bottom == 0:
        return 0
    return top / bottom
 
 
# extra: f1
 
def f1PerClass(matrix):
    # F1 = 2 * precision * recall / (precision + recall)
    precisions = precisionPerClass(matrix)
    recalls = recallPerClass(matrix)
    values = []
    for k in range(len(matrix)):
        p = precisions[k]
        r = recalls[k]
        values.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
    return values
 
 
def macroF1(matrix):
    return average(f1PerClass(matrix))
 
 
# printing the results
 
def printEvaluation(name, yTrue, yPred, classNames=BINARY_NAMES):
    # one function that prints every metric for one algorithm
    classes = getClasses(yTrue, yPred)
    matrix = confusionMatrix(yTrue, yPred, classes)
    precisions = precisionPerClass(matrix)
    recalls = recallPerClass(matrix)
    f1s = f1PerClass(matrix)
 
    print(f"\n{name}")
    printConfusionMatrix(matrix, classes, classNames)
 
    print("  Per class:")
    for i, c in enumerate(classes):
        print(f"    {classNames.get(c, str(c)):<14} precision={precisions[i]:.3f}  "
              f"recall={recalls[i]:.3f}  f1={f1s[i]:.3f}")
 
    print("  Overall:")
    print(f"    accuracy={accuracy(matrix):.3f}")
    print(f"    macro precision={macroPrecision(matrix):.3f}")
    print(f"    balanced accuracy={balancedAccuracy(matrix):.3f}")
    print(f"    MCC={mcc(matrix):.3f}")
 
 
def thresholdTable(yTrue, probs, thresholds, positive=1):
    # only for 2 classes: turn the probability of the positive class into
    # a yes / no using different thresholds, then see how the metrics change
    print("\nDifferent thresholds (predict dynasty if probability >= threshold)")
    print(f"  {'threshold':>9} {'precision':>10} {'recall':>8} {'bal.acc':>8} {'MCC':>7}")
 
    for t in thresholds:
        preds = []
        for p in probs:
            preds.append(1 if p >= t else 0)
 
        matrix = confusionMatrix(yTrue, preds, [0, 1])
        print(f"  {t:>9.1f} {precisionPerClass(matrix)[positive]:>10.3f} "
              f"{recallPerClass(matrix)[positive]:>8.3f} "
              f"{balancedAccuracy(matrix):>8.3f} {mcc(matrix):>7.3f}")
 
 
def explainResults(name, yTrue, yPred, classNames=BINARY_NAMES):
    # prints the results in normal sentences so they are easier to put in the report
    classes = getClasses(yTrue, yPred)
    matrix = confusionMatrix(yTrue, yPred, classes)
    precisions = precisionPerClass(matrix)
    recalls = recallPerClass(matrix)
    total = matrixTotal(matrix)
 
    print(f"\nIn plain words: {name}")
    print(f"  We checked {total:,} rows.")
 
    for i, c in enumerate(classes):
        label = classNames.get(c, str(c))
        real = rowTotal(matrix, i)
        found = matrix[i][i]
        called = columnTotal(matrix, i)
        print(f"  '{label}': {real:,} rows ({real / total * 100:.1f}%) are really this class.")
        print(f"     We found {found:,} of them ({recalls[i] * 100:.1f}%) and missed {real - found:,}.")
        print(f"     We called {called:,} rows '{label}' and {precisions[i] * 100:.1f}% of those were right.")