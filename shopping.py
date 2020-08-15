import csv
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    label = []
    # read data from the csv into pandas data frame
    data_frame = pd.read_csv(filename)
    for row in data_frame.itertuples():
        single_evidence = []
        for i in range(1, 18):
            # all integer values
            if i in [1, 3, 5] + list(range(11, 18)):
                if i == 11:
                    single_evidence.append(get_month(row[i]))
                elif i == 16 or i == 17:
                    if row[i] == "Returning_Visitor" or row[i] == True:
                        single_evidence.append(1)
                    else:
                        single_evidence.append(0)
                else:
                    single_evidence.append(int(row[i]))
            # all floating point values
            else:
                single_evidence.append(float(row[i]))
        # labels
        if row[18] == True:
            label.append(1)
        else:
            label.append(0)
        evidence.append(single_evidence)
    return (evidence, label)

def get_month(month):
    """
    Returns an integer between 0 and 11, corresponding to a month name
    month according to the rule provided: Jan is 0, Feb is 1, ...,
    Dec is 11.
    """
    months = ["Jan", "Feb", "Mar", "Apr", "May", "June",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for i in range(len(months)):
        if month == months[i]:
            return i

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    total_pos = 0
    total_neg = 0
    count_pos = 0
    count_neg = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total_pos += 1
            if predictions[i] == 1:
                count_pos += 1
        else:
            total_neg += 1
            if predictions[i] == 0:
                count_neg += 1
    sensitivity = count_pos / total_pos
    specificity = count_neg / total_neg
    return sensitivity, specificity

if __name__ == "__main__":
    main()
