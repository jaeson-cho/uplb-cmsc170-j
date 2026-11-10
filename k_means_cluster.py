import math
import random
from data_preprocessor import preProcessData
import evaluation

#Constants
FILENAME = "NLE_Winners_2004-2025.csv"
TRAIN_END_YEAR = 2016

FEATURES = [
    "last_name",
    "middle_name",
    "province",
    "city",
    "position"
]

RANDOM_SEED = 42
MAX_ITERATIONS = 100

#labeling
def addLabels(officials):
    local=[]

    for official in officials:
        if (
            official["position"] in ["MAYOR", "VICE MAYOR", "COUNCILOR"]
            and official["city"] != "N/A"
            and official["last_name"] != "N/A"
            and official["year"] != "N/A"
        ):
            local.append(official)
    groups = {}
    for official in local:
        key = (
            official["province"],
            official["city"],
            official["year"],
            official["last_name"]
        )
        if key not in groups:
            groups[key] = set()
        groups[key].add(official["first_name"])

    #Label 1, if more than 2 people share same last name

    for official in local:
        key = (
            official["province"],
            official["city"],
            official["year"],
            official["last_name"]
        )
        if len(groups[key]) >= 2:
            official["label"] = 1
        else:
            official["label"] = 0
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
        vocab[feature] = sorted(
            set(row[feature] for row in trainingSet)
        )
    return vocab

def encodeRows(rows, features, vocab):
    # each row becomes a list of the spot numbers that are 1
    # values that were not in the training data have no spot so we skip them
    encoded = []
    labels = []
    for row in rows:
        spots = []
        for feature in features:
            pair = (feature, row[feature])
            if pair in vocab:
                spots.append(vocab[pair])
        encoded.append(spots)
        labels.append(row["label"])
    return encoded, labels

def generateInitialCentroids(trainingSet, k, size):
    if k > len(trainingSet):
        raise ValueError("K cannot exceed the training set size.")

    selectedRows = random.sample(trainingSet, k)

    centroids = []

    for spots in selectedRows:
        centroid = [0.0] * size
        for s in spots:
            centroid[s] = 1.0
        centroids.append(centroid)
    return centroids

def getNorms(centroids):
    # sum of the squares of each centroid, we need it for the distance formula
    norms = []
    for centroid in centroids:
        total = 0.0
        for value in centroid:
            total += value * value
        norms.append(total)
    return norms

def applyEuclidean(spots, centroid, centroidNorm):
    # centroidNorm - 2 * (sum of the centroid at the row's 1 spots) + (number of 1s)
    dot = 0.0
    for s in spots:
        dot += centroid[s]
    return centroidNorm - 2 * dot + len(spots)

def findNearestCluster(spots, centroids, norms):
    # compare the row with every centroid and return the index of the closest one
    bestCluster = 0
    bestDistance = applyEuclidean(spots, centroids[0], norms[0])
    for i in range(1, len(centroids)):
        distance = applyEuclidean(spots, centroids[i], norms[i])
        if distance < bestDistance:
            bestDistance = distance
            bestCluster = i
    return bestCluster

def applyKmeans(encodedRows, centroids):
    k = len(centroids)
    size = len(centroids[0])
    assignments = [0] * len(encodedRows)
 
    for iteration in range(MAX_ITERATIONS):
        # step 1: assign each row to its nearest centroid
        norms = getNorms(centroids)
        for r in range(len(encodedRows)):
            assignments[r] = findNearestCluster(encodedRows[r], centroids, norms)
 
        # step 2: the new centroid is the average of the rows in the cluster
        # for one-hot rows the average is just (count of 1s in the spot) / (rows in cluster)
        counts = []
        for i in range(k):
            counts.append([0.0] * size)
        clusterSizes = [0] * k
 
        for r in range(len(encodedRows)):
            c = assignments[r]
            clusterSizes[c] += 1
            for s in encodedRows[r]:
                counts[c][s] += 1.0
 
        newCentroids = []
        for i in range(k):
            if clusterSizes[i] == 0:
                # keep the old centroid if nobody joined this cluster
                newCentroids.append(centroids[i])
            else:
                newCentroids.append([value / clusterSizes[i] for value in counts[i]])
 
        # step 3: stop if no centroid moved more than a tiny amount
        converged = True
        for i in range(k):
            for j in range(size):
                if abs(centroids[i][j] - newCentroids[i][j]) > 0.000001:
                    converged = False
                    break
            if not converged:
                break
 
        centroids = newCentroids
        print(f"Iteration {iteration + 1}: cluster sizes = {clusterSizes}")
 
        if converged:
            print(f"K-means converged after {iteration + 1} iterations")
            break
 
    return assignments, centroids


def classifyClusters(assignments, labels, k):
    # each cluster gets the label that most of its training rows have (majority vote)
    clusterLabels = {}
    for i in range(k):
        counts = {}
        for a, label in zip(assignments, labels):
            if a == i:
                counts[label] = counts.get(label, 0) + 1
 
        if len(counts) == 0:
            # empty cluster, we just say label 0
            clusterLabels[i] = 0
        else:
            # the label with the biggest count wins
            clusterLabels[i] = max(counts, key=counts.get)
 
        print(f"Cluster {i}: label counts = {counts} -> classified as {clusterLabels[i]}")
 
    return clusterLabels
 
 
def predictRows(encodedRows, centroids, clusterLabels):
    # a new row gets the label of the cluster whose centroid is nearest
    norms = getNorms(centroids)
    predictions = []
    for spots in encodedRows:
        cluster = findNearestCluster(spots, centroids, norms)
        predictions.append(clusterLabels[cluster])
    return predictions



def main():
    try:
        random.seed(RANDOM_SEED)
        k = int(input("Enter K: "))
 
        if k <= 0:
            print("Error: K must be greater than 0.")
            return
 
        officials = preProcessData(FILENAME)
        if officials is None:
            return
 
        rows = addLabels(officials)
        train, test = splitData(rows)
        print(f"\nRows: {len(rows)} | Training set: {len(train)} | Testing set: {len(test)}")
 
        # turn the rows into numbers
        vocab = generateVocabulary(train, FEATURES)
        x_train, y_train = encodeRows(train, FEATURES, vocab)
        x_test, y_test = encodeRows(test, FEATURES, vocab)
        print(f"Number of features (one-hot spots): {len(vocab)}")
        print(f"Number of clusters (K): {k}")
 
        # run k-means on the training rows only
        centroids = generateInitialCentroids(x_train, k, len(vocab))
        assignments, centroids = applyKmeans(x_train, centroids)
 
        # give each cluster a label then predict the test rows
        print("\nCluster classifications")
        clusterLabels = classifyClusters(assignments, y_train, k)
        predictions = predictRows(x_test, centroids, clusterLabels)
 
        # all the grading is done by evaluation.py
        evaluation.printEvaluation(f"k-means (k = {k})", y_test, predictions)
        evaluation.explainResults(f"k-means (k = {k})", y_test, predictions)
 
    except ValueError as e:
        print(f"Error: {e}")
 
 
if __name__ == "__main__":
    main()