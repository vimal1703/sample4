#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv('pizza.csv')

x=df.drop(['likePizza'],axis=1)
y=df['likePizza']


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x,y)

neigh.predict(x)

pickle.dump(neigh, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
#print(model.predict(sc.transform(np.array([[20,40]]))))


# In[ ]:




