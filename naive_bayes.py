import csv
import math
from data_preprocessor import preProcessData
import evaluation

FILENAME = "NLE_Winners_2004-2025.csv"
TRAIN_END_YEAR = 2016   
FEATURES = ["last_name", "middle_name", "province", "city", "position"]
LOCAL_POSITIONS = ["MAYOR", "VICE MAYOR", "COUNCILOR"]
CLASS_NAMES = {0: "None", 1: "Thin", 2: "Fat", 3: "Obese"}
CLASSES = [0, 1, 2, 3]
ALPHA = 1   # laplace smoothing


# labeling

def addLabels(officials):
    local = []
    for official in officials:
        if (
            official["position"] in LOCAL_POSITIONS
            and official["city"] != "N/A"
            and official["last_name"] != "N/A"
            and official["year"] != "N/A"
        ):
            local.append(official)

    # Collect the first name that share a last name in the same city and year
    groups = {}
    years_by_surname = {}
    for o in local:
        year = int(o["year"])
        province = o["province"]
        city = o["city"]
        lastName = o["last_name"]
        firstName = o["first_name"]
 
        groupKey = (province, city, lastName, year)
        historyKey = (province, city, lastName)
 
        # make an empty set the first time we see a key
        if groupKey not in groups:
            groups[groupKey] = set()
        groups[groupKey].add(firstName)
 
        if historyKey not in years_by_surname:
            years_by_surname[historyKey] = set()
        years_by_surname[historyKey].add(year)
 
    # the first year each last name showed up in each city
    first_year = {}
    for key, years in years_by_surname.items():
        first_year[key] = min(years)
 
    for o in local:
        year = int(o["year"])
        province = o["province"]
        city = o["city"]
        lastName = o["last_name"]
 
        count = len(groups[(province, city, lastName, year)])
        earliestYear = first_year[(province, city, lastName)]
 
        if count >= 4:
            o["label"] = 3      # obese: 4 or more share the last name this year
        elif count >= 2:
            o["label"] = 2      # fat: 2 or 3 share the last name this year
        elif earliestYear < year:
            o["label"] = 1      # thin: the last name was already in this city before
        else:
            o["label"] = 0      # none
 
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
    x = []
    y = []
    for r in rows:
        x.append([r[f] for f in features])
        y.append(r["label"])
    return x, y


# Naive Bayes

def trainNaiveBayes(X, y):
    # A dictionary that stores all the counts we need
    numFeatures = len(X[0])
    classCount = {}
 
    # valueCount[c][i][v] = how many rows of class c had value v in feature i
    valueCount = {}
    for c in CLASSES:
        classCount[c] = 0
        valueCount[c] = []
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
    score = model["classCount"][c] / model["total"]
 
    for i in range(len(row)):
        # how many class c rows had this exact value in feature i (0 if never seen)
        count = model["valueCount"][c][i].get(row[i], 0)
 
        # laplace smoothing so one unseen value does not make the whole product 0
        # the +1 in the bottom is an extra spot for values not in the training data
        top = count + ALPHA
        bottom = model["classCount"][c] + ALPHA * (model["vocabSize"][i] + 1)
 
        # Nave Bayes: we assume the features are independent so multiply
        # we add the logs instead because multiplying many small numbers may cause underflow
        score = score * (top / bottom)
 
    return score
 
 
def predictProba(model, row):
    # get the score of every class (4 now, it was 2 before) for this row
    scores = {}
    total = 0.0
    for c in CLASSES:
        scores[c] = logScore(model, row, c)
        total += scores[c]
 
    # the scores are not probabilities yet so we need to normalize them
    # P(class | row) = score / (sum of all the scores), so all 4 probabilities add up to 1
    probs = {}
    for c in CLASSES:
        probs[c] = scores[c] / total
    return probs
 
 
def predict(model, row):
    # the predicted class is the one with the highest probability
    probs = predictProba(model, row)
    bestClass = CLASSES[0]
    for c in CLASSES:
        if probs[c] > probs[bestClass]:
            bestClass = c
    return bestClass

def countClasses(labels):
    # how many rows of each class, just for printing
    counts = {}
    for c in CLASSES:
        counts[c] = 0
    for label in labels:
        counts[label] += 1
    return counts


def main():
    officials = preProcessData(FILENAME)
    if officials is None:
        return
 
    rows = addLabels(officials)
    train, test = splitData(rows)
    print(f"rows: {len(rows)} | train: {len(train)} | test: {len(test)}")
 
    X_train, y_train = getFeatures(train, FEATURES)
    X_test, y_test = getFeatures(test, FEATURES)
    
     # how many rows of each type we have
    trainCounts = countClasses(y_train)
    testCounts = countClasses(y_test)
    print("\nRows per type:")
    for c in CLASSES:
        print(f"  {CLASS_NAMES[c]:<6} train={trainCounts[c]:>7,} ({trainCounts[c] / len(y_train) * 100:4.1f}%)"
              f"   test={testCounts[c]:>7,} ({testCounts[c] / len(y_test) * 100:4.1f}%)")
    
    # baseline to compare with
    majority = max(trainCounts, key=trainCounts.get)
    evaluation.printEvaluation(f"baseline: always predict {CLASS_NAMES[majority]}", y_test,
                               [majority] * len(y_test), classNames=CLASS_NAMES)
    
    # naive bayes with all the features
    model = trainNaiveBayes(X_train, y_train)
    allprobs = [predictProba(model, row) for row in X_test]
    preds = []
    for probs in allprobs:
        # pick the class with the highest probability
        preds.append(max(probs, key=probs.get))
    evaluation.printEvaluation("naive bayes (all features)", y_test, preds, classNames=CLASS_NAMES)
 
    # naive bayes with only the last name to see which feature matters the most
    X_train2, _ = getFeatures(train, ["last_name"])
    X_test2, _ = getFeatures(test, ["last_name"])
    model2 = trainNaiveBayes(X_train2, y_train)
    preds2 = [predict(model2, row) for row in X_test2]
    evaluation.printEvaluation("naive bayes (last name only)", y_test, preds2, classNames=CLASS_NAMES)
 
    print("\nExample predictions with the probability of each type:")
    for i in range(5):
        r = test[i]
        probs = allprobs[i]
        print(f"  {r['position']}, {r['city']}, {r['province']}, {r['year']}: "
              f"real={CLASS_NAMES[y_test[i]]}, predicted={CLASS_NAMES[preds[i]]}")
        line = "    "
        for c in CLASSES:
            line += f"{CLASS_NAMES[c]}={probs[c] * 100:5.1f}%  "
        print(line)
 
    # the results in normal sentences for the report
    evaluation.explainResults("naive bayes (all features)", y_test, preds, classNames=CLASS_NAMES)
    evaluation.explainResults("naive bayes (last name only)", y_test, preds2, classNames=CLASS_NAMES)
 
 
if __name__ == "__main__":
    main()