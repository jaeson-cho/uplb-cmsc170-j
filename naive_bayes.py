import csv
import math
from data_preprocessor import preProcessData

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


# Evaluation Matrix

def confusionMatrix(yTrue, yPred):
    tp = fp = tn = fn = 0
    for t, p in zip(yTrue, yPred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 0 and p == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn
 
 
def getMetrics(yTrue, yPred):
    tp, fp, tn, fn = confusionMatrix(yTrue, yPred)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return tp, fp, tn, fn, accuracy, precision, recall, f1
 
 
def printMetrics(name, yTrue, yPred):
    tp, fp, tn, fn, acc, prec, rec, f1 = getMetrics(yTrue, yPred)
    print(f"{name}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  accuracy={acc:.3f} precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")


def explanations(y_test, preds_all, preds_name_only):
    total = len(y_test)
    dynasties = sum(y_test)
    non_dynasties = total - dynasties
    pct_dynasties = (dynasties / total) * 100

    print(f"\n Evaluated {total:,} local election winners after 2016")
    print(f" Only {dynasties:,} ({pct_dynasties:.1f}%) were part of an election dynasty.")
    print(f" The vast majority ({non_dynasties:,}) were NOT dynasties")

    # All Features
    tp_all, fp_all, tn_all, fn_all = confusionMatrix(y_test, preds_all)
    prec_all = (tp_all / (tp_all + fp_all)) * 100 if (tp_all + fp_all) > 0 else 0
    rec_all = (tp_all / (tp_all + fn_all)) * 100 if (tp_all + fn_all) > 0 else 0

    print("\nUSING ALL INFORMATION (name, city, province, position)")
    print(f" The model found {tp_all:,} dynastic officials ({rec_all:.1f}% of all dynasties).")
    print(f" It completely overlooked {fn_all:,} dynasties")
    print(f" It wrongly accused {fp_all:,} winners of being in a dynasty")
    print(f" When it flags someone as a dynasty. It is {prec_all:.1f}% of the time.")

    # Last Names only
    tp_name, fp_name, tn_name, fn_name = confusionMatrix(y_test, preds_name_only)
    prec_name = (tp_name / (tp_name + fp_name)) * 100 if (tp_name + fp_name) > 0 else 0
    rec_name = (tp_name / (tp_name + fn_name)) * 100 if (tp_name + fn_name) > 0 else 0

    print("\nUSING ONLY LAST NAME")
    print(f" When it flags someone, it is right {prec_name:.1f}% of the time.")
    print(f" It only catches {tp_name:,} dynasties ({rec_name:.1f}%) and misses {fn_name:,} ({100 - rec_name:.1f}%) because it doesn't recognize newer or less-prominent political surnames")
# main

def main():
    officials = preProcessData(FILENAME)
    if officials is None:
        return
 
    rows = addLabels(officials)
    train, test = splitData(rows)
    print(f"rows: {len(rows)} | train: {len(train)} | test: {len(test)}")
 
    dynasties = sum(r["label"] for r in test)
    print(f"dynasty rows in test set: {dynasties / len(test):.1%}")
    print()
 
    X_train, y_train = getFeatures(train, FEATURES)
    X_test, y_test = getFeatures(test, FEATURES)
 
    # baselines to compare with
    printMetrics("always predict not dynasty", y_test, [0] * len(y_test))
    printMetrics("always predict dynasty", y_test, [1] * len(y_test))
 
    # naive bayes with all the features
    model = trainNaiveBayes(X_train, y_train)
    probs = [predictProba(model, row) for row in X_test]
    preds = [1 if p >= 0.5 else 0 for p in probs]
    printMetrics("naive bayes (all features)", y_test, preds)
 
    # try only the last name to see which feature matters most
    X_train2, _ = getFeatures(train, ["last_name"])
    X_test2, _ = getFeatures(test, ["last_name"])
    model2 = trainNaiveBayes(X_train2, y_train)
    preds2 = [predict(model2, row) for row in X_test2]
    printMetrics("naive bayes (last name only)", y_test, preds2)
 
    # try different thresholds on the dynasty probability
    print()
    print("different thresholds (all features)")
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        p = [1 if pr >= t else 0 for pr in probs]
        _, _, _, _, acc, prec, rec, f1 = getMetrics(y_test, p)
        print(f"  threshold={t}: precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")

    explanations(y_test, preds, preds2)

    
main()
