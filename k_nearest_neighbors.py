import csv
import math
from data_preprocessor import preProcessData

FILENAME = "NLE_Winners_2004-2025.csv"
TRAIN_END_YEAR = 2016
FEATURES = [
    "last_name",
    "middle_name",
    "province",
    "city",
    "position"
]
LOCAL_POSITIONS = ["MAYOR", "VICE MAYOR", "COUNCILOR"]

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

def generateVocabulary(trainingSet, features):
    vocab = {}

    for feature in features:
        vocab[feature] = sorted(set(row[feature] for row in trainingSet))

    return vocab

def getFeatures(rows, features, vocab):
    x = []
    y = []

    for row in rows:
        encodedRow = []

        for feature in features:
            value = row[feature]

            for category in vocab[feature]:
                encodedRow.append(1.0 if value == category else 0.0)

        x.append(encodedRow)
        y.append(row["label"])

    return x, y

def applyMinowski(row1, row2, p):
    distance = 0

    for i in range(len(row1)):
        distance += abs(row1[i] - row2[i]) ** p

    return distance ** (1/p)

def getNeighbors(trainingSet, testRow, k, p):
    distances = []

    for index, trainRow in enumerate(trainingSet):
        distance = applyMinowski(testRow, testRow["features"], p)

        distances.append([
            distance, 
            index, 
            trainRow["label"]
        ])

    distances.sort()

    return distances[:k] 

def predict(neighbors, k):
    dynastyCount = 0

    for neighbor in neighbors:
        if neighbor[2] == 1:
            dynastyCount += 1

    if dynastyCount > k/2:
        return 1

    return 0

def applyKNN(trainingSet, testSet, k, p):
    predictions = []

    for testRow in testSet:
        neighbors = getNeighbors(trainingSet, testRow, k, p)

        prediction = predict(neighbors, k)
        predictions.append(prediction)

    return predictions

def confusionMatrix(yTrue, yPred):
    tp = fp = tn = fn = 0

    for t, p in zip(yTrue, yPred):
        if t == 1 and p==1:
            tp += 1
        elif t == 0 and p==1:
            fp += 1
        elif t == 0 and p == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn

def getMetrics(yTrue, yPred):
    tp, fp, tn, fn, = confusionMatrix(yTrue, yPred)

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision+recall)>0 else 0
    )

    return tp, fp, tn, fn, accuracy, precision, recall, f1

def printMetrics(name, yTrue, yPred):
    tp, fp, tn, fn, acc, prec, rec, f1 = getMetrics(yTrue, yPred)

    print(name)
    print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")
    print(
        f"    accuracy={acc:.3f}  "
        f"precision={prec:.3f}  "
        f"recall={rec:.3f}  "
        f"f1={f1:.3f}"    
    )

#main
def main():
    try:
        k = int(input("Enter k: "))
        p = float(input("Enter p: "))

        if k <= 0:
            print("Error: K must be greater than 0.")
            return

        if p < 1:
            print("Error, P must be at least 1.")
            return

        officials = preProcessData(FILENAME)

        if officials is None:
            return

        rows = addLabels(officials)

        #TESTING
        print("First year:", min(row["year"] for row in rows))
        print("Last year:", max(row["year"] for row in rows))
        print("Years found:", sorted(set(row["year"] for row in rows)))


        train, test = splitData(rows)

        print(f"\nRows: {len(rows)}")
        print(f"Training set: {len(train)}")
        print(f"Testing set: {len(test)}")

        if not train or not test:
            print("Error: Training or Testing set is empty.")
            return

        if k > len(train):
            print("Error: K cannot exceed training set size.")
            return

        dynasties = sum(row["label"] for row in test)

        print(
            f"Dynasty rows in test set: "
            f"{dynasties / len(test):.1%}"
        )

        vocab = generateVocabulary(train, FEATURES)
        print("\nFinished generating vocab...")

        x_train, y_train = getFeatures(train, FEATURES, vocab)
        x_test, y_test = getFeatures(test, FEATURES, vocab)

        print(f"Number of features: {len(x_train[0])}")
        print(f"Number of Neighbors (k): {k}")
        print(f"Minowski parameter (p): {p}")

        trainingSet = []

        for features, label in zip(x_train, y_train):
            trainingSet.append({
                "features": features,
                "label": label
            })

        predictions = applyKNN(
            trainingSet,
            x_test,
            k,
            p
        )

        print("\nKNN Classification results")

        printMetrics("K Nearest Neighbors (All Features)", y_test, predictions)

        print()
        printMetrics("Always predict not dynasty", y_test, [0]*len(y_test))

        print()
        printMetrics("Always predict dynasty", y_test, [1]*len(y_test))

    except ValueError as e:
        print(f"Error: {e}")

main()