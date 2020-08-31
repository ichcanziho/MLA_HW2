import pandas as pd
import random
import numpy as np

class VicImplementation ():

    def __init__(self,dataFile,outputName,class_outs=[0,1],n_outs=2,colsToRemove=[]):
        # dataFile = route to the file to be loaded into a pandas dataFrame in CSV
        # outputName = the name of the class column
        # class_outs = the name of the output classes
        # n_outs = number of csv to generate
        # colsToRemove = Optional parameter to remove certain cols given their names

        self.route = dataFile
        self.n_outs = n_outs
        self.n_classes = len(class_outs)
        self.class_values = class_outs
        self.data = pd.read_csv(dataFile)
        self.numberOfElements = len(self.data)
        self.data.sort_values(by=[outputName],inplace=True)
        for col in colsToRemove:
            self.data.drop([col], axis=1, inplace=True)

        self.data.drop([outputName], axis=1,inplace=True)
        self.outputName = outputName

    # wrapper function of the dirichlet function
    def makeSplits(self):
        # n_classes = the number of classe to be mapped
        # n_outs = the number of csv to generate
        self.splits = np.random.dirichlet(np.ones(self.n_classes), size=self.n_outs)

    def makeCategoricalOutput(self):
        # generate self.splits
        self.makeSplits()
        j=1
        # for each split in splits
        for split in self.splits:
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
            dataOut.to_csv('data_outputs/split_{}_nc_{}.csv'.format(j,self.n_classes),index=False)
            j+=1

vic = VicImplementation('data/Data_15s_30r.csv','score_change',class_outs=['a','b'],n_outs=2,colsToRemove=['fingerprint','minutia'])


vic.makeCategoricalOutput()