import math
import random
from data_preprocessor import preProcessData

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

def getFeatures(rows, features, vocab):
    x=[]
    y=[]
    print(f"feature {features}")
    print(f"row {rows}")
    for row in rows:
        encodedRow = []
        for feature in features:
            value = row[feature]
            for category in vocab[feature]:
                encodedRow.append(
                    1.0 if value == category else 0.0
                )
            
        x.append(encodedRow)
        y.append(row["label"])
    return x,y

def generateInitialCentroids(trainingSet, k):
    if k > len(trainingSet):
        raise ValueError("K cannot exceed the training set size.")

    selectedRows = random.sample(trainingSet, k)

    centroids = []

    for row in selectedRows:
        centroids.append(row.copy())

    return centroids
def applyEuclidean(centroid, row):
    distance = 0
    for i in range(len(centroid)):
        distance += (centroid[i] - row[i]) ** 2
    return math.sqrt(distance)

def applyKmeans(trainingSet, centroids):
    k = len(centroids)
    maxIterations= 100

    for iteration in range(maxIterations):
        #assigns each row to nearest centroid
        for row in trainingSet:
            distances = []
            for centroid in centroids:
                distance = applyEuclidean(centroid, row["features"])
                distances.append(distance)
            cluster_index = distances.index(min(distances))
            row['cluster'] = cluster_index
        newCentroids =[]

        for cluster_index in range(k):
            clusterPoints = [
                row for row in trainingSet
                if row["cluster"] == cluster_index
            ]

            #keep if empty
            if not clusterPoints:
                newCentroids.append(centroids[cluster_index].copy())
                continue
            numFeatures = len(centroids[cluster_index])

            newCentroid = []

            for feature_index in range(numFeatures):
                total = sum(
                    row["features"][feature_index]
                    for row in clusterPoints
                )

                average = total/len(clusterPoints)
                newCentroid.append(average)
            newCentroids.append(newCentroid)
        #converged flag
        converged = True
        for i in range(k):
            if any(
                abs(centroids[i][j] - newCentroids[i][j]) > 0.000001
                for j in range(len(centroids[i]))
            ):
                converged = False
                break
        centroids = newCentroids

        print(f"Iteration {iteration + 1}")

        for i, centroid in enumerate(centroids):
            print(f"Centroid {i}: {centroid[:5]}...")
        print("=========")

        if converged:
            print(f"\nK-means has converged within {iteration + 1} iterations")
            break
        print(f"iterating.. {iteration + 1}")
    return trainingSet, centroids


def classifyClusters(trainingSet, k):
    clusterLabels = {}
    for cluster_index in range(k):
        clusterPoints = [
            row for row in trainingSet
            if row["cluster"] == cluster_index
        ]

        dynastyCount = sum(
            row["label"] for row in clusterPoints
        )

        nonDynastyCount = len(clusterPoints) - dynastyCount

        clusterLabels[cluster_index] = (
            1 if dynastyCount > nonDynastyCount else 0
        )

        print(
            f"Cluster {cluster_index}: "
            f"Dynasty= {dynastyCount} "     
            f"Not Dynasty= {nonDynastyCount}"
            f"Classification={clusterLabels[cluster_index]}"
        )

    return clusterLabels



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

        print(f"\nRows: {len(rows)}")
        print(f"Training set: {len(train)}")
        print(f"Testing set: {len(test)}")

        dynasties = sum(row["label"] for row in test)
        # print(f"Dynasty rows in test set: \n{dynasties / len(test):.1}")
        vocab = generateVocabulary(train, FEATURES)
        print("Finished generating vocab...")
        x_train, y_train = getFeatures(train, FEATURES, vocab)
        print("Acquired training features...")
        x_test, y_test = getFeatures(test,FEATURES, vocab)
        print("Acquired test features...")
        print(f"\nNumber of features: {len(x_train[0])}")
        print(f"Number of clusters (K): {k}")

        trainingSet = []

        for features, label in zip(x_train, y_train):
            trainingSet.append({
                "features": features,
                "label": label
            })
        print("Appended training set...")
        #Generate centroids
        centroids = generateInitialCentroids(x_train, k)
        print("\nInitial Centroids Generated...")
        trainingSet, centroids = applyKmeans(trainingSet, centroids)

        print("\n Final centorids calculated.")
        # for centroid in centroids:
        #     print(f"Centroid: {centroid}")
        print("\nCluster Classifications")
        clusterLabels = classifyClusters(
            trainingSet,
            k
        )
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()