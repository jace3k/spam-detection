import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
# użyć wilcoxon
# https://github.com/w4k2/benchmark_datasets
FEATURES = 10  # zbadać dla różnych ilości cech
FOLDS = 5
DATASET_PATH = 'datasets/spam_sms.csv'


def get_values_and_labels(ds):
    return ds.values[:, 1], ds.values[:, 0]


def preprocess_data(dataset):
    X, y = get_values_and_labels(dataset)
    print('Dataset shape: {}'.format(X.shape))
    print('Vectorizing data..')
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X, y).toarray()
    print('Vectorized shape: {}'.format(X.shape))

    print('Feature reduction..')
    pca = PCA(n_components=FEATURES, copy=False)
    X = pca.fit_transform(X)

    print('Scaling..')
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    print('Shape: {}'.format(X.shape))

    return X, y


def predict(X, y, clf, k=6):
    skf = model_selection.StratifiedKFold(n_splits=k)
    scores = np.zeros(k)
    for fold, (train, test) in enumerate(skf.split(X, y)):
        print('Fold: {}/{}'.format(fold + 1, k))
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        print('-- fitting..')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        score = metrics.accuracy_score(y_test, y_pred)  # f_score, precision, recall

        print('-- accuracy: {}'.format(score))
        scores[fold] = score

    return np.mean(scores)


def main():
    knn = neighbors.KNeighborsClassifier()
    dataset = pd.read_csv(DATASET_PATH, encoding='latin-1')

    X, y = preprocess_data(dataset)
    result = predict(X, y, knn, FOLDS)
    print('Accuracy: {:2f}%'.format(result * 100))


if __name__ == '__main__':
    main()

# dataset - mniejszosciowa musi być pozytywna
# zbiór : 5574 elementy, 747 spam, 4827 ham.
