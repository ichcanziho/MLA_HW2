import pandas as pd
import random
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
#vic = VicImplementation('data/test.csv','output',class_outs=[0,1],n_outs=3)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

class VicImplementation ():

    def __init__(self,dataFile,outputName,class_outs=[0,1],n_outs=2,colsToRemove=[]):
        # dataFile = route to the file to be loaded into a pandas dataFrame in CSV
        # outputName = the name of the class column
        # class_outs = the name of the output classes
        # n_outs = number of csv to generate
        # colsToRemove = Optional parameter to remove certain cols given their names
        self.rocHistory = []
        self.bestRoc=0
        self.bestClassifier = "NA"
        self.kfolds = 2
        self.route = dataFile
        self.n_outs = n_outs
        self.n_classes = len(class_outs)
        self.class_values = class_outs
        self.data = pd.read_csv(dataFile)
        self.data = self.data.fillna(self.data.mean())

        self.numberOfElements = len(self.data)
        self.data.sort_values(by=[outputName],inplace=True)
        for col in colsToRemove:
            self.data.drop([col], axis=1, inplace=True)

        self.data.drop([outputName], axis=1,inplace=True)
        self.outputName = outputName
        self.classifiers = []

    def addClassifier(self,*args):
        self.classifiers+=args
        print(self.classifiers)
    # wrapper function of the dirichlet function
    def makeSplits(self):
        # n_classes = the number of classe to be mapped
        # n_outs = the number of csv to generate
        self.splits = np.random.dirichlet(np.ones(self.n_classes), size=self.n_outs)

    def makeNewSplits(self):
        grupos = self.n_outs
        #totalDatos = self.numberOfElements
        minPartition = 10
        step = int((100 - minPartition) / grupos)

        split = [[i/100,(100-i)/100]for i in range(minPartition, 100-minPartition, step)]
        for s in split:
            print(s)
        self.splits = split
    def makeCategoricalOutput(self):
        # generate self.splits
        self.makeSplits()
        print(self.splits)
        print("----")
        self.makeNewSplits()

        j=1
        # for each split in splits
        for split in self.splits:
            print("Iteration {}/{}".format(j,self.n_outs))
            partial = 0
            output=[]
            i=0
            for value in split[:-1]:
                # convert the dirichlet probability into a portion of the number of elements
                cut = int(value*self.numberOfElements)
                # convert the cut number into a list of each class in class values
                output+=[self.class_values[i]]*cut
                partial+=cut
                i+=1
            # the last element is the complement of the sum of the previous elements
            cut = self.numberOfElements - partial
            output += [self.class_values[i]] * cut
            # make the dataframe output
            dataOut = self.data.copy()
            dataOut[self.outputName] = output
            # save it to a csv file
            currentRoc, currentClassifier = self.obtainVIC(dataOut)
            currentRoc = round(currentRoc,4)
            self.rocHistory.append(currentRoc)
            if currentRoc>self.bestRoc:
                self.bestRoc=currentRoc
                self.bestClassifier = currentClassifier
                print("Best result ROC {} Classifier {}".format(self.bestRoc,self.bestClassifier))
                dataOut.to_csv('data_outputs/best_split.csv',index=False)
            j+=1
        print(self.rocHistory)
        self.plotAucHistory()


    def plotCurve(self,new_actual_class, new_pred_class, current_class, model_name, multi=False):
        ns_probs = [0 for _ in range(len(new_actual_class))]
        ns_fpr, ns_tpr, _ = roc_curve(new_actual_class, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(new_actual_class, new_pred_class)
        # plot the roc curve for the model
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        plt.plot(lr_fpr, lr_tpr, marker='.', label=model_name)
        # axis labels
        classes = set(new_actual_class)
        if multi:
            plt.xlabel('False Positive Rate Class:' + str(current_class))
        else:
            plt.xlabel('False Positive Rate')

        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()

    def roc_auc_score_multiclass(self,y_true, y_pred, model_name, curve=False, average="macro"):
        classes = set(y_true)
        roc_auc_dict = {}
        for current_class in classes:
            # creating a list of all the classes except the current class
            other_class = [x for x in classes if x != current_class]
            # marking the current class as 1 and all other classes as 0
            new_actual_class = [0 if x in other_class else 1 for x in y_true]
            new_pred_class = [0 if x in other_class else 1 for x in y_pred]
            # using the sklearn metrics method to calculate the roc_auc_score
            roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
            roc_auc_dict[current_class] = roc_auc
            if curve:
                if len(classes) == 2 and current_class == 1:
                    self.plotCurve(new_actual_class, new_pred_class, current_class, model_name)
                if len(classes) != 2:
                    self.plotCurve(new_actual_class, new_pred_class, current_class, model_name, multi=True)

        avg = sum(roc_auc_dict.values()) / len(roc_auc_dict)

        return round(avg,4), roc_auc_dict

    def setKfolds(self,kfolds):
        self.kfolds = kfolds

    def obtainVIC(self,DF):
        kFolds = self.kfolds
        x_DF = DF.iloc[:, :-1].values
        y_DF = DF.iloc[:, -1]
        V = []
        # Define the list of classifiers to use
        clfs = self.classifiers

        for clf in clfs:
            kfold = StratifiedKFold(n_splits=kFolds, shuffle=True)
            aucTotal = 0
            name = clf.__class__.__name__
            n_fold=1
            for train_index, test_index in kfold.split(x_DF, y_DF):
                print("Fold number:",n_fold)
                n_fold+=1
                X_train, X_test = x_DF[train_index], x_DF[test_index]
                y_train, y_test = y_DF[train_index], y_DF[test_index]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                aucV, _ = self.roc_auc_score_multiclass(y_test, y_pred, name)
                aucTotal += aucV
            real_auc = aucTotal / kFolds
            print(name,real_auc)
            V.append((real_auc,name))
        #print("all")
        #print(V)
        #print("----------------")
        V.sort(reverse=True)
        print(V)
        return V[0]

    def plotAucHistory(self):
        bestAUC = self.rocHistory
        partitions = range(1,len(bestAUC)+1)
        bestAUCIndex = bestAUC.index(max(bestAUC))
        worstAUCIndex = bestAUC.index(min(bestAUC))

        # Plotting bar graphs with certain aesthetic alterations for better visualization.
        y_pos = np.arange(len(partitions))
        fig, ax = plt.subplots()
        barlist = ax.bar(y_pos, bestAUC, align='center', alpha=0.7, color=['green'])
        barlist[bestAUCIndex].set_color('b')
        barlist[worstAUCIndex].set_color('r')

        for i, v in enumerate(bestAUC):
            if (i == bestAUCIndex):
                ax.text(i - 2.8, v + .001, str(v), color='blue', fontweight='bold', size=12)
            if (i == worstAUCIndex):
                ax.text(i - 2.1, v + .001, str(v), color='red', fontweight='bold', size=12)

        plt.xticks(y_pos, partitions, rotation='vertical')
        plt.ylabel('ROC-AUC', fontweight='bold', size=12, labelpad=10)
        plt.ylim((min(bestAUC) * 0.99, max(bestAUC) * 1.01))
        plt.xlabel('Partition Number', fontweight='bold', size=12, labelpad=15)
        plt.title('Best Performance of Classifiers Among Each Partition', fontweight='bold', size=15)

        plt.show()

vic = VicImplementation('data/Data_15s_30r.csv','score_change',class_outs=[0,1],n_outs=5,colsToRemove=['fingerprint','minutia'])
vic.setKfolds(3)
vic.addClassifier(SVC(gamma='auto'))#, RandomForestClassifier())#, GaussianNB(), LinearDiscriminantAnalysis(), MLPClassifier())
vic.makeCategoricalOutput()