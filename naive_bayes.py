import csv
import math
from data_preprocessor import preProcessData
import evaluation

FILENAME = "NLE_Winners_2004-2025.csv"
TRAIN_END_YEAR = 2016   
FEATURES = ["last_name", "middle_name", "province", "city", "position"]
LOCAL_POSITIONS = ["MAYOR", "VICE MAYOR", "COUNCILOR"]
ALPHA = 1   # laplace smoothing


# labeling

def addLabels(officials):
    local = []
    for o in officials:
        if o["position"] in LOCAL_POSITIONS and o["city"] != "N/A" and o["last_name"] != "N/A" and o["year"] != "N/A":
            local.append(o)

    # Collect the first name that share a last name in the same city and year
    groups = {}
    for o in local:
        key = (o["province"], o["city"], o["year"], o["last_name"])
        if key not in groups:
            groups[key] = set()
        groups[key].add(o["first_name"])

    # Label 1 (dynasty) if 2 or more different people share the last name
    for o in local:
        key = (o["province"], o["city"], o["year"], o["last_name"])
        if len(groups[key]) >= 2:
            o["label"] = 1
        else:
            o["label"] = 0
 
    return local


def splitData(rows):
    train = []
    test = []
    for r in rows:
        if r["year"] <= TRAIN_END_YEAR:
            train.append(r)
        else:
            test.append(r)
    return train, test
 
 
def getFeatures(rows, features):
    X = []
    y = []
    for r in rows:
        X.append([r[f] for f in features])
        y.append(r["label"])
    return X, y


# Naive Bayes

def trainNaiveBayes(X, y):
    # A dictionary that stores all the counts we need
    numFeatures = len(X[0])
    classCount = {0: 0, 1: 0}
 
    # valueCount[c][i][v] = how many rows of class c had value v in feature i
    valueCount = {0: [], 1: []}
    for c in (0, 1):
        for i in range(numFeatures):
            valueCount[c].append({})
 
    # vocab[i] = all the different values feature i has in the training data
    vocab = []
    for i in range(numFeatures):
        vocab.append(set())
 
    # training is only counting
    for row, c in zip(X, y):
        classCount[c] += 1
        for i in range(numFeatures):
            v = row[i]
            # get(v, 0) gives 0 the first time we see this value
            valueCount[c][i][v] = valueCount[c][i].get(v, 0) + 1
            vocab[i].add(v)
 
    # we only need the size of each vocab later for the laplace smoothing
    vocabSize = []
    for i in range(numFeatures):
        vocabSize.append(len(vocab[i]))
 
    model = {
        "classCount": classCount,
        "valueCount": valueCount,
        "vocabSize": vocabSize,
        "total": len(X)
    }
    return model
 
 
def logScore(model, row, c):
    # start with the prior P(c) = (rows in class c) / (all rows)
    score = math.log(model["classCount"][c] / model["total"])
 
    for i in range(len(row)):
        # how many class c rows had this exact value in feature i (0 if never seen)
        count = model["valueCount"][c][i].get(row[i], 0)
 
        # laplace smoothing so one unseen value does not make the whole product 0
        # the +1 in the bottom is an extra spot for values not in the training data
        top = count + ALPHA
        bottom = model["classCount"][c] + ALPHA * (model["vocabSize"][i] + 1)
 
        # Nave Bayes: we assume the features are independent so multiply
        # we add the logs instead because multiplying many small numbers may cause underflow
        score += math.log(top / bottom)
 
    return score
 
 
def predictProba(model, row):
    # get the log score of both classes for this row
    s0 = logScore(model, row, 0)
    s1 = logScore(model, row, 1)
 
    # the scores are not probabilities yet, We need normalization
    # subtract the bigger one first so exp() does not underflow to 0
    m = max(s0, s1)
    e0 = math.exp(s0 - m)
    e1 = math.exp(s1 - m)
 
    # P(dynasty | row) = e1 / (e0 + e1), Percentage for the report
    return e1 / (e0 + e1)
 
 
def predict(model, row, threshold=0.5):
    # call it a dynasty only if the probability reaches the threshold
    if predictProba(model, row) >= threshold:
        return 1
    return 0


def main():
    officials = preProcessData(FILENAME)
    if officials is None:
        return
 
    rows = addLabels(officials)
    train, test = splitData(rows)
    print(f"rows: {len(rows)} | train: {len(train)} | test: {len(test)}")
 
    X_train, y_train = getFeatures(train, FEATURES)
    X_test, y_test = getFeatures(test, FEATURES)
 
    # baselines to compare with, they do not use any algorithm
    evaluation.printEvaluation("baseline: always predict not dynasty", y_test, [0] * len(y_test))
    evaluation.printEvaluation("baseline: always predict dynasty", y_test, [1] * len(y_test))
 
    # naive bayes with all the features
    model = trainNaiveBayes(X_train, y_train)
    probs = [predictProba(model, row) for row in X_test]
    preds = [1 if p >= 0.5 else 0 for p in probs]
    evaluation.printEvaluation("naive bayes (all features)", y_test, preds)
 
    # naive bayes with only the last name to see which feature matters the most
    X_train2, _ = getFeatures(train, ["last_name"])
    X_test2, _ = getFeatures(test, ["last_name"])
    model2 = trainNaiveBayes(X_train2, y_train)
    preds2 = [predict(model2, row) for row in X_test2]
    evaluation.printEvaluation("naive bayes (last name only)", y_test, preds2)
 
    # try different thresholds on the dynasty probability
    evaluation.thresholdTable(y_test, probs, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
 
    # the results in normal sentences for the report
    evaluation.explainResults("naive bayes (all features)", y_test, preds)
    evaluation.explainResults("naive bayes (last name only)", y_test, preds2)
 
 
if __name__ == "__main__":
    main()