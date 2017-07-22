
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
'''
Python 3.5.2
developer mauricio munoz, contact: yemauricio@gmail.com
'''


# In[2]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")


# In[3]:


# Importing dataset and split it according of the class i.e. class D or S
dataset = pd.read_csv(".\SalesKaggle3\SalesKaggle3.csv")
dataset_copy =  dataset.copy()


# In[4]:


ClassD_product = dataset_copy[dataset_copy.MarketingType == "D"]
ClassD_product = ClassD_product[["SKU_number","ReleaseNumber","New_Release_Flag","PriceReg","ItemCount","LowUserPrice","LowNetPrice", "SoldFlag"]]
ClassS_product = dataset_copy[dataset_copy.MarketingType == "S"]
ClassS_product = ClassS_product[["SKU_number","ReleaseNumber","New_Release_Flag","PriceReg","ItemCount","LowUserPrice","LowNetPrice", "SoldFlag"]]
ClassD_product_forprediction = ClassD_product[ClassD_product.SoldFlag.isnull()]
ClassD_product_fortrain = ClassD_product[ClassD_product.SoldFlag.isnull()== False]
ClassS_product_forprediction = ClassS_product[ClassS_product.SoldFlag.isnull()]
ClassS_product_fortrain = ClassS_product[ClassS_product.SoldFlag.isnull()== False]


# In[5]:


# Data for training and testing for class D products
X_train_D = ClassD_product_fortrain[["ReleaseNumber","New_Release_Flag","PriceReg","ItemCount","LowUserPrice","LowNetPrice"]]
Y_train_D = ClassD_product_fortrain[["SoldFlag"]]
# test data is the data that will be forecasted
X_test_D = ClassD_product_forprediction[["ReleaseNumber","New_Release_Flag","PriceReg","ItemCount","LowUserPrice","LowNetPrice"]]


# In[6]:


# Data for training and testing for class S products
X_train_S = ClassS_product_fortrain[["ReleaseNumber","New_Release_Flag","PriceReg","ItemCount","LowUserPrice","LowNetPrice"]]
Y_train_S = ClassS_product_fortrain[["SoldFlag"]]
X_test_S = ClassS_product_forprediction[["ReleaseNumber","New_Release_Flag","PriceReg","ItemCount","LowUserPrice","LowNetPrice"]]


# In[7]:


# Logistic regression
logistic_R = LogisticRegression()


# In[9]:


# The training step is developed using 90% of the data, the resting 10% is used to evaluate the Error
logistic_D_train = logistic_R.fit(X_train_D[:31608],Y_train_D[:31608])


# In[11]:


prediction_D_train = logistic_D_train.predict(X_train_D[31608:])
prediction_D_train = pd.DataFrame(prediction_D_train)
prediction_D_train.columns = ["Prediction D"]


# In[12]:


# Error level of the logistic classifier
accuracy = 0
for i in range(0, len(prediction_D_train)):
    if Y_train_D["SoldFlag"].ix[31608+i] == prediction_D_train["Prediction D"].ix[i]:
        accuracy += 1
accuracy/len(prediction_D_train)*100


# In[14]:


# Forecasting for the target data and presenting 
prediction_D_test = logistic_D_train.predict(X_test_D)
prediction_D_test = pd.DataFrame(prediction_D_test)
prediction_D_test.columns = ["Prediction SoldFlag D"]
ClassD_product_forprediction = ClassD_product_forprediction.reset_index()
ClassD_product_forprediction["SoldFlag"] = prediction_D_test["Prediction SoldFlag D"]
ClassD_product_forprediction = ClassD_product_forprediction.set_index("index")
ClassD_product_forprediction


# In[15]:


# Products that would have 1.0 flag 
ClassD_product_forprediction[ClassD_product_forprediction["SoldFlag"]>0]


# In[16]:


# Logistic training for S Class items
logistic_S_train = logistic_R.fit(X_train_S[:36790],Y_train_S[:36790])


# In[19]:


prediction_S_train = logistic_S_train.predict(X_train_S[36790:])
prediction_S_train = pd.DataFrame(prediction_S_train)
prediction_S_train.columns = ["Prediction S"]


# In[20]:


# Error level of the classifier
accuracy_S = 0
for i in range(0, len(Y_train_S[36790:])):
    if Y_train_S["SoldFlag"].ix[71909+i] == prediction_S_train["Prediction S"].ix[i]:
        accuracy_S += 1
accuracy_S/len(prediction_S_train)*100


# In[21]:


# Forecasting for the target data and presenting Results
prediction_S_test = logistic_S_train .predict(X_test_S)
prediction_S_test= pd.DataFrame(prediction_S_test)
prediction_S_test.columns = ["Prediction S"]
ClassS_product_forprediction = ClassS_product_forprediction.reset_index()
ClassS_product_forprediction["SoldFlag"] = prediction_S_test["Prediction S"]
ClassS_product_forprediction = ClassS_product_forprediction.set_index("index")
ClassS_product_forprediction


# In[22]:


# Products that would have 1.0 flag 
ClassS_product_forprediction[ClassS_product_forprediction["SoldFlag"]>0]


# In[ ]:




