import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def norm_arr(arr):
    mean = arr.mean()
    std = arr.std()

    normalized = (arr - mean) / std
    return normalized


def norm_med_arr(arr):
    median = arr.median()

    normalized = arr - median
    return normalized


def norm_df(df, usestd=True):
    result = df.copy()

    for feature in df.columns:
        if usestd:
            result[feature] = norm_arr(result[feature])
        else:
            result[feature] = norm_med_arr(result[feature])

    return result


def stratified_split(y, proportion = 0.8):
    y = np.array(y)

    train_inds = np.zeros(len(y), dtype=bool)
    test_inds = np.zeros(len(y), dtype=bool)

    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y == value)[0]
        np.random.shuffle(value_inds)

        n = int(proportion * len(value_inds))

        train_inds[value_inds[:n]] = True
        test_inds[value_inds[n:]] = True

    return train_inds, test_inds


def accuracy(y_test, y_pred):
    return 1 - sum(abs(y_test - y_pred)/len(y_test))


def accuracy_bcr(y_test, y_pred):
    class0 = y_test == 0
    class1 = y_test == 1
    return (accuracy(y_test[class0], y_pred[class0]) + accuracy(y_test[class1], y_pred[class1])) / 2


def CV(df, classifier, nfold, norm='std'):
    result = []
    result_bcr = []
    col = len(df.columns) - 1
    y = df['class']
    for i in range(nfold):
        train, test = stratified_split(y)

        if norm in ['std', 'median']:
            X_train = norm_df(df.iloc[train, 0:col], norm == 'std')
            X_test = norm_df(df.iloc[test, 0:col], norm == 'std')
        else:
            X_train = df.iloc[train, 0:col]
            X_test = df.iloc[test, 0:col]

        y_train = y[train]
        y_test = y[test]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        result.append(accuracy(y_test, y_pred))
        result_bcr.append(accuracy_bcr(y_test, y_pred))

    return result, result_bcr


url = "data/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
cols = [['preg', 'plas', 'pres', 'skin', 'class'], ['test', 'mass', 'pedi', 'age', 'class']]

# Pregnancies - Number of times pregnant - Numeric
# Glucose - Plasma glucose concentration a 2 hours in an oral glucose tolerance test - Numeric
# BloodPressure - Diastolic blood pressure (mm Hg) - Numeric
# SkinThickness - Triceps skin fold thickness (mm) - Numeric
# Insulin - 2-Hour serum insulin (mu U/ml) - Numeric
# BMI - Body mass index (weight in kg/(height in m)^2) - Numeric
# DiabetesPedigreeFunction - Diabetes pedigree function - Numeric
# Age - Age (years) - Numeric
# Outcome - Class variable (0 or 1) - Nu;meric


for c in cols:
    print('cols: ' + str(c))
    df = pd.read_csv(url, names=names)[c]

    lr_acc = []
    lr_acc_bcr = []
    rf_acc = []
    rf_acc_bcr = []

    result, result_bcr = CV(df, LogisticRegression(), 100, 'none')
    lr_acc.append(np.mean(result))
    lr_acc_bcr.append(np.mean(result_bcr))

    result, result_bcr = CV(df, RandomForestClassifier(), 100, 'none')
    rf_acc.append(np.mean(result))
    rf_acc_bcr.append(np.mean(result_bcr))

    result, result_bcr = CV(df, LogisticRegression(), 100, 'std')
    lr_acc.append(np.mean(result))
    lr_acc_bcr.append(np.mean(result_bcr))

    result, result_bcr = CV(df, RandomForestClassifier(), 100, 'std')
    rf_acc.append(np.mean(result))
    rf_acc_bcr.append(np.mean(result_bcr))

    result, result_bcr = CV(df, LogisticRegression(), 100, 'median')
    lr_acc.append(np.mean(result))
    lr_acc_bcr.append(np.mean(result_bcr))

    result, result_bcr = CV(df, RandomForestClassifier(), 100, 'median')
    rf_acc.append(np.mean(result))
    rf_acc_bcr.append(np.mean(result_bcr))

    res = pd.DataFrame(
        data = {'LogisticRegression': lr_acc, 'LogisticRegression BCR': lr_acc_bcr, 'RandomForestClassifier': rf_acc, 'RandomForestClassifier BCR': rf_acc_bcr},
        index = ['un-norm', 'norm', 'median']
    )
    print(res)
    print()
