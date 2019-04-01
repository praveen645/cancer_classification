from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import pickle

cancer_ds=pd.read_csv('cervical_cancer_replaced.csv')
print (cancer_ds.shape)

feature=cancer_ds.iloc[:,:9]
target=cancer_ds.iloc[:,9]


gb=GaussianNB()
gb.fit(feature,target)


file = open('cancer.pickle','wb')
print(file)
pickle.dump(gb,file)

print(" *************************************************************************************** ")

print(" *************************************************************************************** ")

file = open('cancer.pickle','rb')
gb1 = pickle.load(file)

print(gb1.predict([[34,2,1,1,0,1,0,1,1]]))
