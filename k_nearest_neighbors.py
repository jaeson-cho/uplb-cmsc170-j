import csv
import math
from data_preprocessor import preProcessData
import evaluation

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
INDEXED_FEATURES = ["last_name", "middle_name", "city"]
CLASS_NAMES = {0: "None", 1: "Thin", 2: "Fat", 3: "Obese"}
CLASSES = [0, 1, 2, 3]

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

def getTrainFeatures(rows, features):
    # each training row becomes a tuple of its values
    values = []
    labels = []
    for r in rows:
        values.append(tuple(r[f] for f in features))
        labels.append(r["label"])
    return values, labels

def getTestFeatures(rows, features, trainValues):
    # same as above but a value that never showed up in training becomes None
    known = []
    for i in range(len(features)):
        known.append(set())
    for row in trainValues:
        for i in range(len(features)):
            known[i].add(row[i])
 
    values = []
    labels = []
    for r in rows:
        encoded = []
        for i in range(len(features)):
            value = r[features[i]]
            if value in known[i]:
                encoded.append(value)
            else:
                encoded.append(None)
        values.append(tuple(encoded))
        labels.append(r["label"])
    return values, labels

def buildIndex(trainValues, features):
    # index[(feature number, value)] = list of the training rows that have this value
    # this lets us find rows with the same last name or city without checking every row
    index = {}
    for f in range(len(features)):
        if features[f] not in INDEXED_FEATURES:
            continue
        for rowNumber in range(len(trainValues)):
            key = (f, trainValues[rowNumber][f])
            if key not in index:
                index[key] = []
            index[key].append(rowNumber)
    return index


def countMatches(trainRow, testRow):
    # how many features have the same value in both rows
    matches = 0
    for i in range(len(trainRow)):
        if testRow[i] is not None and trainRow[i] == testRow[i]:
            matches += 1
    return matches

def applyMinkowski(matches, testRow, p):
    knownValues = 0
    for value in testRow:
        if value is not None:
            knownValues += 1
    differences = knownValues + len(testRow) - 2 * matches
    return differences ** (1 / p)

def getNeighbors(trainValues, trainLabels, index, testRow, k, p, features):
    # count the matching indexed features (last name, middle name, city) of every training row that shares at least one of them with the test row
    
    sharedCount = {}
    for f in range(len(features)):
        if features[f] not in INDEXED_FEATURES or testRow[f] is None:
            continue
        for rowNumber in index.get((f, testRow[f]), []):
            sharedCount[rowNumber] = sharedCount.get(rowNumber, 0) + 1

    # a row with none of those shared can only match the other features (province, position)
    notIndexed = len(features) - len(INDEXED_FEATURES)
    
    # add the matches of the other features
    candidates = []
    good = 0
    for rowNumber, shared in sharedCount.items():
        matches = shared
        for f in range(len(features)):
            if features[f] not in INDEXED_FEATURES:
                if testRow[f] is not None and trainValues[rowNumber][f] == testRow[f]:
                    matches += 1
        candidates.append((-matches, rowNumber))
        if matches > notIndexed:
            good += 1
            
    # if we have k rows that beat anything, we are done. Else we have to check every training row
    if good < k:
        candidates = []
        for rowNumber in range(len(trainValues)):
            matches = countMatches(trainValues[rowNumber], testRow)
            candidates.append((-matches, rowNumber))
    
    # Closest first
    
    candidates.sort()
 
    neighbors = []
    for negativeMatches, rowNumber in candidates[:k]:
        distance = applyMinkowski(-negativeMatches, testRow, p)
        neighbors.append([distance, rowNumber, trainLabels[rowNumber]])
    return neighbors

def getVotes(neighbors):
    # the share of the k neighbors that voted for each type
    votes = {}
    for c in CLASSES:
        votes[c] = 0
    for neighbor in neighbors:
        votes[neighbor[2]] += 1
 
    shares = {}
    for c in CLASSES:
        shares[c] = votes[c] / len(neighbors)
    return shares

def predict(neighbors):
    # the predicted type is the one with the most votes
    shares = getVotes(neighbors)
    mostVotes = max(shares.values())
 
    # if some types are tied, the tied type of the closest neighbor wins
    for neighbor in neighbors:
        if shares[neighbor[2]] == mostVotes:
            return neighbor[2]

    return 0

def applyKNN(trainValues, trainLabels, testValues, k, p, features):
    index = buildIndex(trainValues, features)
    
    predictions = []
    allShares = []
    for number, testRow in enumerate(testValues):
        neighbors = getNeighbors(trainValues, trainLabels, index, testRow, k, p, features)
        predictions.append(predict(neighbors))
        allShares.append(getVotes(neighbors))
        
        if (number + 1) % 10000 == 0:
            print(f"  classified {number + 1:,} of {len(testValues):,} test rows...")

    return predictions, allShares

def countClasses(labels):
    # how many rows of each class, just for printing
    counts = {}
    for c in CLASSES:
        counts[c] = 0
    for label in labels:
        counts[label] += 1
    return counts

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
        
        trainValues, y_train = getTrainFeatures(train, FEATURES)
        testValues, y_test = getTestFeatures(test, FEATURES, trainValues)
        
        trainCounts = countClasses(y_train)
        testCounts = countClasses(y_test)
        print("\nRows per type:")
        for c in CLASSES:
            print(f"  {CLASS_NAMES[c]:<6} train={trainCounts[c]:>7,} ({trainCounts[c] / len(y_train) * 100:4.1f}%)"
                  f"   test={testCounts[c]:>7,} ({testCounts[c] / len(y_test) * 100:4.1f}%)")
 
        print(f"\nNumber of Neighbors (k): {k}")
        print(f"Minkowski parameter (p): {p}")
        print("Classifying the test rows...")
        
        predictions, allShares = applyKNN(trainValues, y_train, testValues, k, p, FEATURES)
        
        # baseline to compare with, it always answers with the most common training type
        majority = max(trainCounts, key=trainCounts.get)
        evaluation.printEvaluation(f"baseline: always predict {CLASS_NAMES[majority]}", y_test,
                                   [majority] * len(y_test), classNames=CLASS_NAMES)
 
        name = f"K Nearest Neighbors (k={k}, p={p})"
        evaluation.printEvaluation(name, y_test, predictions, classNames=CLASS_NAMES)
        
        # knn gives the share of neighbors that voted for each type
        print("\nExample predictions with the share of neighbors that voted for each type:")
        for i in range(5):
            r = test[i]
            shares = allShares[i]
            print(f"  {r['position']}, {r['city']}, {r['province']}, {r['year']}: "
                  f"real={CLASS_NAMES[y_test[i]]}, predicted={CLASS_NAMES[predictions[i]]}")
            line = "    "
            for c in CLASSES:
                line += f"{CLASS_NAMES[c]}={shares[c] * 100:5.1f}%  "
            print(line)
 
        # the results in normal sentences for the report
        evaluation.explainResults(name, y_test, predictions, classNames=CLASS_NAMES)
 
    except ValueError as e:
        print(f"Error: {e}")
        

main()