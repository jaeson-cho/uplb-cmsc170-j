import math
import random
from data_preprocessor import preProcessData
import evaluation
from collections import Counter, defaultdict
#Constants
FILENAME = "NLE_Winners_2004-2025.csv"
TRAIN_END_YEAR = 2016
LOCAL_POSITIONS = {"MAYOR", "VICE MAYOR", "COUNCILOR"}
CLASS_NAMES = {0: "None" , 1: "Thin", 2:"Fat", 3:"Obese"}
FEATURES = [
    "last_name",
    "middle_name",
    "province",
    "city",
    "position"
]

RANDOM_SEED = 42
MAX_ITERATIONS = 100
TOLERANCE = 0.000001

#labeling
def addLabels(officials):
    local=[]

    for official in officials:
        if (
            official["position"] in LOCAL_POSITIONS
            and official["city"] != "N/A"
            and official["last_name"] != "N/A"
            and official["year"] != "N/A"
        ):
            local.append(official)
    groups = defaultdict(set)
    years_by_surname = defaultdict(set)
    for o in local:
        year = int(o["year"])
        
        province = o["province"]
        city = o["city"]
        lastName = o["last_name"]
        firstName = o["first_name"]
        
        locationSurname = (province, city, lastName)
        groupKey = (province, city, lastName, year)

        if groupKey not in groups:
            groups[groupKey] = set()

        groups[groupKey].add(firstName)

        if locationSurname not in years_by_surname:
            years_by_surname[locationSurname] = set()

        years_by_surname[locationSurname].add(year)
    first_year = {key: min(years) for key, years in years_by_surname.items()}

    #Label 1, if more than 2 people share same last name

    for o in local:
        year = int(o["year"])
        province = o["province"]
        city = o["city"]
        lastName = o["last_name"]

        groupKey = (province,city, lastName, year)
        historyKey = (province, city, lastName)

        firstNames = groups[groupKey]
        count = len(firstNames)

        earliestYear = first_year[historyKey]

        if count >= 4:
            #obese
            o["label"] = 3

        elif count >= 2:
            #fat
            o["label"] = 2

        elif earliestYear < year:
            #thin
            o["label"] = 1
        else:
            #none
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
        #set for none dupes
        uniqueValues = set()
        for row in trainingSet:
            value = row[feature]
            uniqueValues.add(value)
        #sorting
        sortedValues = sorted(uniqueValues)
        #assign ids
        for value in sortedValues:
            pair = (feature, value)
            index = len(vocab)
            vocab[pair] = index
    return vocab

def encodeRows(rows, features, vocab):
    # each row becomes a list of the spot numbers that are 1
    # values that were not in the training data have no spot so we skip them
    encoded = []
    labels = []
    for row in rows:
        spots = []
        for feature in features:
            index = vocab.get((feature, row[feature]))
            if index is not None:
                spots.append(index)
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
            squaredValue = value * value
            total += squaredValue
        norms.append(total)
    return norms

def applyEuclidean(spots, centroid, centroidNorm):
    # centroidNorm - 2 * (sum of the centroid at the row's 1 spots) + (number of 1s)
    dot = 0.0
    for s in spots:
        value = centroid[s]
        dot += value
    numberOfSpots = len(spots)
    distance = centroidNorm - (2 * dot) + numberOfSpots
    return distance

def findNearestCluster(spots, centroids, norms):
    # compare the row with every centroid and return the index of the closest one
    bestCluster = 0

    bestDistance = applyEuclidean(spots, centroids[0], norms[0])

    for i in range(1, len(centroids)):
        centroid = centroids[i]
        centroidNorm = norms[i]
        distance = applyEuclidean(
            spots, 
            centroid, 
            centroidNorm
        )

        if distance < bestDistance:
            bestDistance = distance
            bestCluster = i
    return bestCluster

def assignRows(encodedRows, centroids):
    #squared norms of centroids
    norms = getNorms(centroids)

    #store assigned cluster per row
    assignments = []

    for spots in encodedRows:
        nearestCluster = findNearestCluster(
            spots,
            centroids,
            norms
        )
        assignments.append(nearestCluster)
    return assignments

    
def applyKmeans(encodedRows, centroids):
    k = len(centroids)
    size = len(centroids[0])
 
    for iteration in range(MAX_ITERATIONS):
        #assign each row to nearest centroid
        assignments = assignRows(encodedRows, centroids)
        #counts per cluster init
        counts = []

        for i in range(k):
            clusterCounts = [0.0] * size
            counts.append(clusterCounts)
        #Track num of rows
        clusterSizes = [0] * k
        #active features per row
        for row in range(len(encodedRows)):
            spots = encodedRows[row]
            cluster = assignments[row]
            clusterSizes[cluster] += 1
       #each active fes pern the row, add 1 to that cluster's count for that fee. Hot-encoding 
            for spot in spots:
                counts[cluster][spot] += 1.0
        newCentroids = []

        for i in range(k):
            #if cluster empty, retain old
            if clusterSizes[i] == 0:
                newCentroid = centroids[i][:]
                newCentroids.append(newCentroid)
            else:
                newCentroid = []
                for value in counts[i]:
                    average = value / clusterSizes[i]
                    newCentroid.append(average)
                newCentroids.append(newCentroid)
        
        converged = True
        for i in range(k):
            oldCentroid = centroids[i]
            newCentroid = newCentroids[i]

            for j in range(size):
                oldValue = oldCentroid[j]
                newValue = newCentroid[j]

                diff = abs(oldValue-newValue)

                if diff > TOLERANCE:
                    converged = False
                    break
            if not converged:
                break
        
        #update if not converged
        centroids = newCentroids
        # For debugging. Comment below since printing significantly slows down processes.
        # print(f"Iteration {iteration+1}: Cluster sizes = {clusterSizes}")
        if(converged):
            print(f"K-means converged after {iteration+1} iterations")
            break
    finalAssignments = assignRows(encodedRows, centroids)
    return finalAssignments, centroids       


def classifyClusters(assignments, labels, k):
    # each cluster gets the label that most of its training rows have (majority vote)
    counts = []
    #init label counts per cluster
    for i in range(k):
        clusterCounts = {}
        counts.append(clusterCounts)
    
    # Count training labels per cluster
    for i in range(len(assignments)):
        cluster = assignments[i]
        label = labels[i]

        if label not in counts[cluster]:
            counts[cluster][label] = 0
        counts[cluster][label] += 1
    clusterLabels = {}

    for i in range(k):
        clusterCounts = counts[i]
        #empty cluster defaults
        bestLabel = 0
        highestCount = 0
        #check each possible dynasty classification
        for label in CLASS_NAMES:
            if label in clusterCounts:
                currentCount = clusterCounts[label]
            else:
                currentCount= 0
            
            if currentCount > highestCount:
                highestCount = currentCount
                bestLabel = label
            
            elif currentCount == highestCount and currentCount > 0:
                if label < bestLabel:
                    bestLabel = label
        clusterLabels[i] = bestLabel
        classification = CLASS_NAMES[bestLabel]

        print(
            f"Cluster {i}: "
            f"counts={dict(sorted(clusterCounts.items()))} "
            f"-> {classification} ({bestLabel})"
        )
    return clusterLabels
 
def predictRows(encodedRows, centroids, clusterLabels):
    # a new row gets the label of the cluster whose centroid is nearest
    assignments = assignRows(encodedRows, centroids)

    predictions = []

    for cluster in assignments:
        predictedLabel = clusterLabels[cluster]
        predictions.append(predictedLabel)

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
        name = f"K-means (k={k})"
        # all the grading is done by evaluation.py
        evaluation.printEvaluation(name, y_test, predictions, classNames=CLASS_NAMES)
        evaluation.explainResults(name, y_test, predictions, classNames=CLASS_NAMES)
 
    except ValueError as e:
        print(f"Error: {e}")
 
 
if __name__ == "__main__":
    main()