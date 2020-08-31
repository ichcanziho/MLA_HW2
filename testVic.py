import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

#from sklearn.model_selection import cross_val_predict
#from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import auc
#from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from sklearn.base import BaseEstimator, ClassifierMixin

iris = datasets.load_iris()
X = iris.data
y = iris.target

#----------

clfs = [RandomForestClassifier(),GaussianNB(),LinearDiscriminantAnalysis(),MLPClassifier(),SVC(probability=True)]

y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]
for clf in clfs:
    print("----")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
    classifier = OneVsRestClassifier(clf)
    y_score = classifier.fit(X_train, y_train)
    y_prob = classifier.predict_proba(X_test)
    macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",average="macro")
    weighted_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",average="weighted")
    print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
          "(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo))

V=[]
for clf in clfs:
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=30)
    aucTotal = 0
    for train_index, test_index in kfold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = OneVsRestClassifier(clf)
        clf.fit(X_train,y_train)
        y_prob = clf.predict_proba(X_test)
        aucV = roc_auc_score(y_test, y_prob, multi_class="ovo",average="macro")
        aucTotal+=aucV
    V.append(aucTotal/5)

print(V)
#Sort to obtain the highest AUC
V.sort(reverse = True)
