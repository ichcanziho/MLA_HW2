import pandas as pd
import random
import numpy as np
'''
random.seed(5)
output = [random.randint(-300,300)/100 for i in range(500)]
col1 = [random.randint(0,600)/100 for i in range(500)]
col2 = [random.randint(0,600)/100 for i in range(500)]

df = pd.DataFrame({"a":col1,"b":col2,"output":output})

print(df.head())

df.to_csv("data/test.csv",index=False)
'''


class VicImplementation ():
    def __init__(self,dataFile,outputName,class_outs=[0,1],n_outs=2):
        self.route = dataFile
        self.n_outs = n_outs
        self.n_classes = len(class_outs)
        self.class_values = class_outs
        self.data = pd.read_csv(dataFile)
        self.numberOfElements = len(self.data)
        self.data.sort_values(by=[outputName],inplace=True)
        self.data.drop([outputName], axis=1,inplace=True)
        self.outputName = outputName

    def makeSplits(self):
        self.splits = np.random.dirichlet(np.ones(self.n_classes), size=self.n_outs)

    def makeCategoricalOutput(self):
        self.makeSplits()
        j=1
        for split in self.splits:
            partial = 0
            output=[]
            i=0
            for value in split[:-1]:
                cut = int(value*self.numberOfElements)
                output+=[self.class_values[i]]*cut
                partial+=cut
                i+=1
            cut = self.numberOfElements - partial
            output += [self.class_values[i]] * cut

            dataOut = self.data.copy()
            dataOut[self.outputName] = output

            dataOut.to_csv('data_outputs/split_{}_nc_{}.csv'.format(j,self.n_classes),index=False)
            j+=1

vic = VicImplementation('data/Data_15s_30r.csv','score_change',class_outs=['a','b','c'],n_outs=1)


vic.makeCategoricalOutput()