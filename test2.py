from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import pandas as pd
from roc_multi import roc_auc_score_multiclass
from sklearn.metrics import roc_auc_score
DF = pd.read_csv('data_outputs/test_split_1_nc_2.csv')

x_DF = DF.iloc[:, :-1].values
y_DF = DF.iloc[:, -1]

V = []

# Define the list of classifiers to use
clfs = [SVC(), RandomForestClassifier(), GaussianNB(), LinearDiscriminantAnalysis(), MLPClassifier()]

# Create the kfolds with 5 folds. Obtain the AUC for each classifier.
for clf in clfs:
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    aucTotal = 0
    name = clf.__class__.__name__
    au2=0
    for train_index, test_index in kfold.split(x_DF, y_DF):
        X_train, X_test = x_DF[train_index], x_DF[test_index]
        y_train, y_test = y_DF[train_index], y_DF[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        aucV,_ = roc_auc_score_multiclass(y_test,y_pred,name)
        aucTotal += aucV
        au2 += roc_auc_score(y_test, y_pred)
    real_auc = aucTotal / 5
    print(name)
    print(real_auc,au2/5)
    V.append(real_auc)

# Sort to obtain the highest AUC
V.sort(reverse=True)