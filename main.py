import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# TODO: wilcoxon
# https://github.com/w4k2/benchmark_datasets

FEATURES = 5
FOLDS = 5
DATASET_PATH = 'datasets/spam_sms.csv'


def get_values_and_labels(ds):
    return ds.values[:, 1], ds.values[:, 0]


def preprocess_data(dataset):
    X, y = get_values_and_labels(dataset)

    print('Transform labels..')
    y = LabelEncoder().fit_transform(y)

    print('Dataset shape: {}'.format(X.shape))
    print('Vectorizing data..')
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X, y).toarray()
    print('Vectorized shape: {}'.format(X.shape))

    # df = pd.DataFrame(X, y)
    # print(df)
    # df.to_csv('datasets/sms_spam_all_features.csv')

    print('Feature reduction..')
    pca = PCA(n_components=FEATURES, copy=False)
    X = pca.fit_transform(X)

    # df = pd.DataFrame(X, y)
    # print(df)
    # df.to_csv('datasets/sms_spam_{}_features.csv'.format(FEATURES))

    print('Scaling..')
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    print('Shape: {}'.format(X.shape))

    return X, y


def predict(X, y, clf, k=6):
    skf = model_selection.StratifiedKFold(n_splits=k)
    f1_scores = []
    precision_scores = []
    recall_scores = []
    for fold, (train, test) in enumerate(skf.split(X, y)):
        print('Fold: {}/{}'.format(fold + 1, k))
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        print('-- fitting..')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        f1_score = metrics.f1_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)

        f1_scores.append(f1_score)
        precision_scores.append(precision)
        recall_scores.append(recall)

    return np.mean(f1_scores), np.mean(precision_scores), np.mean(recall_scores)


def main():
    knn = neighbors.KNeighborsClassifier()
    dataset = pd.read_csv(DATASET_PATH, encoding='latin-1')

    X, y = preprocess_data(dataset)
    result = predict(X, y, knn, FOLDS)
    print('F1 Score: {:2f}'.format(result[0]))
    print('Precision: {:2f}'.format(result[1]))
    print('Recall: {:2f}'.format(result[2]))
    el = [FEATURES]
    el.extend(result)
    print('{},{},{},{}'.format(*el))


if __name__ == '__main__':
    main()
