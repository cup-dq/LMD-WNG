from collections import Counter
from LMDWNG import LMDWNG
import numpy as np
import pandas as pd
dataset = pd.read_csv('ar3.csv', encoding='utf-8', delimiter=",")
X_train=dataset.iloc[:,1:-1]
y_train=dataset.iloc[:,-1]
clf=LMDWNG()
X_resampled,y=clf.fit(X_train,y_train,n_nbor=3,distance='Euclidean')

