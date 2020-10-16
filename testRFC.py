import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
#data = pd.read_csv('data/DataTest.csv')
#data.drop(['fingerprint', 'minutia','Unnamed: 125','Unnamed: 126'], axis=1, inplace=True)
#data = data.fillna(data.mean())
#data.to_csv('data/cleanTest.csv',index=False)
#print(data.head(10))

data = pd.read_csv('data/data8_3_30.csv')
#data.drop(['fingerprint', 'minutia',"Unnamed: 125" , "Unnamed: 126"], axis=1, inplace=True)
data.drop(['fingerprint', 'minutia'], axis=1, inplace=True)
data = data.fillna(data.mean())
print(data)

y = data['score_change'].tolist()


def mapOutputs(y,cuts):
    classes = list(range(0, len(cuts) + 1))
    classes= classes[::-1]
    cuts.append(len(y))
    y_sorted = y.copy()
    y_sorted.sort()
    cuts = cuts[::-1]
    pivots = [y_sorted[n-1] for n in cuts]
    outpus=[5]*len(y)
    for pivot, label in zip(pivots, classes):
        i=0
        for value in y:
            if value <= pivot:
                outpus[i] = label
            i+=1
    return outpus

def vic(data):
    kFolds = 5
    x_DF = data.iloc[:, :-1].values
    y_DF = data.iloc[:, -1]
    print(y_DF)
    V = []
    # Define the list of classifiers to use
    clfs = [SVC(gamma='auto'), RandomForestClassifier(),GaussianNB(), LinearDiscriminantAnalysis(), MLPClassifier()]

    for clf in clfs:
        kfold = StratifiedKFold(n_splits=kFolds, shuffle=True)
        aucTotal = 0
        name = clf.__class__.__name__
        n_fold=1
        for train_index, test_index in kfold.split(x_DF, y_DF):

            n_fold+=1
            X_train, X_test = x_DF[train_index], x_DF[test_index]
            y_train, y_test = y_DF[train_index], y_DF[test_index]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            #aucV, _ = self.roc_auc_score_multiclass(y_test, y_pred, name)
            aucV = roc_auc_score(y_test, y_pred) #average="macro"
            print("Fold number:", n_fold,aucV)
            aucTotal += aucV
        real_auc = aucTotal / kFolds
        print(name,real_auc)
        V.append((real_auc,name))

#mapOutputs(y,[500])

outputs=[1 if i>=0 else 0 for i in y]
#print(outputs)
#y2 = pd.DataFrame({'class':outputs})
data['class'] = outputs
print(data)
vic(data)