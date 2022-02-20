#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rcParams['figure.dpi'] = 300


# In[4]:


dataset = pd.read_csv("SCC_vs_Healthy.csv")
dataset.head()


# In[5]:


dataset["HGNC"].value_counts()


# In[6]:


X = dataset.iloc[:,1:]
X.head()


# In[7]:


Y=dataset.iloc[:,:1]
Y.head()


# In[8]:


from sklearn.decomposition import PCA
sklearn_pca = PCA(n_components=2)
print(X.shape)
PCs = sklearn_pca.fit_transform(X)
print(PCs.shape)


# In[9]:


dataset_transform = pd.DataFrame(PCs,columns=['PC1','PC2'])
dataset_transform = pd.concat([dataset_transform,Y],axis=1)
fig, axes = plt.subplots(figsize=(4,4))
sns.set_style("whitegrid")
sns.scatterplot(x='PC1',y='PC2',data = dataset_transform,hue='HGNC',s=60)


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)


# In[11]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[12]:


y_pred = classifier.predict(X_test)


# In[13]:


y_pred


# In[14]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[15]:


sns.heatmap(cm,annot=True,fmt='d',cmap="gray")


# In[16]:


print(classifier.intercept_)
print(classifier.coef_)


# In[17]:


print(classifier.classes_)


# In[18]:


coeff_df = pd.DataFrame(classifier.coef_.T, X.columns, columns=['Coefficient_SCC_vs_Healthy'])


# In[19]:


coeff_df.to_excel('parameters_SCC_vs_Healthy.xlsx')


# In[20]:


import xgboost as xgb


# In[21]:


xgb_mod=xgb.XGBClassifier(random_state=42) # build classifier
xgb_mod=xgb_mod.fit(X_train,y_train.values.ravel()) 


# In[22]:


y_pred = xgb_mod.predict(X_test)

# Performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[23]:


import shap
explainer = shap.TreeExplainer(xgb_mod)
shap_values = explainer.shap_values(X)
expected_value = explainer.expected_value

############## visualizations #############
# Generate summary dot plot
shap.summary_plot(shap_values, X,title="SHAP summary plot") 


# In[24]:


shap_values


# In[25]:


shap.summary_plot(shap_values, X,plot_type="bar",max_display=14) 

