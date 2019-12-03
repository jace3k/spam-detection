"""
Prosty eksperyment klasyfikacji.
"""
import numpy as np
import pandas as pd
from sklearn import base
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors, naive_bayes, tree, svm, neural_network
from sklearn import preprocessing

LABELS_LAST = True
LABELS_FIRST = False


def get_datasets():
    return [
        (pd.read_csv('datasets/spam_email.data'), LABELS_LAST),
        (pd.read_csv('datasets/spam_sms.csv', encoding='latin-1'), LABELS_FIRST)
    ]


def get_values_and_labels(ds):
    labels_last = ds[1]
    data = ds[0]
    if labels_last:
        return data.values[:, :-2], data.values[:, -1]
    return data.values[:, 1:], data.values[:, 0]


datasets = get_datasets()

clfs = {
    "kNN": neighbors.KNeighborsClassifier(),
    "GNB": naive_bayes.GaussianNB(),
    "DTC": tree.DecisionTreeClassifier(),
    "SVC": svm.SVC(gamma='scale'),
    "MLP": neural_network.MLPClassifier(),
}

k = 5

scores = np.zeros((len(datasets), len(clfs), k))

for did, dataset in enumerate(datasets):
    X, y = get_values_and_labels(dataset)

    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    scaled_X = scaler.transform(X)

    skf = model_selection.StratifiedKFold(n_splits=k)
    for fold, (train, test) in enumerate(skf.split(scaled_X, y)):
        X_train, X_test = scaled_X[train], scaled_X[test]
        y_train, y_test = y[train], y[test]

        for cid, clfn in enumerate(clfs):
            clf = base.clone(clfs[clfn])

            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            score = metrics.accuracy_score(y_test, y_pred)

            scores[did, cid, fold] = score


print(np.mean(scores, axis=2))
