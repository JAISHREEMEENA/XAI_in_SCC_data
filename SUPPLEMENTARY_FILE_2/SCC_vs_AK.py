#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rcParams['figure.dpi'] = 300


# In[2]:


dataset = pd.read_csv("SCC_vs_AK_feature_data.csv")
dataset.head()


# In[3]:


dataset["Condition"].value_counts()


# In[4]:


X = dataset.iloc[:,1:13]
X.head()


# In[5]:


Y=dataset.iloc[:,:1]
Y.head()


# In[6]:


from sklearn.decomposition import PCA
sklearn_pca = PCA(n_components=2)
print(X.shape)
PCs = sklearn_pca.fit_transform(X)
print(PCs.shape)


# In[7]:


dataset_transform = pd.DataFrame(PCs,columns=['PC1','PC2'])
dataset_transform = pd.concat([dataset_transform,Y],axis=1)
fig, axes = plt.subplots(figsize=(4,4))
sns.set_style("whitegrid")
sns.scatterplot(x='PC1',y='PC2',data = dataset_transform,hue='Condition',s=60)


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)


# In[9]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[10]:


y_pred = classifier.predict(X_test)


# In[11]:


y_pred


# In[12]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[13]:


sns.heatmap(cm,annot=True,fmt='d',cmap="gray")


# In[14]:


print(classifier.intercept_)
print(classifier.coef_)


# In[15]:


print(classifier.classes_)


# In[16]:


coeff_df = pd.DataFrame(classifier.coef_.T, X.columns, columns=['Coefficient_SCC_vs_AK'])


# In[17]:


coeff_df.to_excel('parameters_SCC_vs_AK.xlsx')


# In[18]:


import xgboost as xgb


# In[19]:


xgb_mod=xgb.XGBClassifier(random_state=42) # build classifier
xgb_mod=xgb_mod.fit(X_train,y_train.values.ravel()) 


# In[20]:


y_pred = xgb_mod.predict(X_test)

# Performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[21]:


annot_kws={'fontsize':25}
sns.heatmap(cm,annot=True,fmt='d',cmap="gray", annot_kws=annot_kws)


# In[22]:


import shap
explainer = shap.TreeExplainer(xgb_mod)
shap_values = explainer.shap_values(X)
expected_value = explainer.expected_value

############## visualizations #############
# Generate summary dot plot
shap.summary_plot(shap_values, X,title="SHAP summary plot") 


# In[23]:


shap_values


# In[24]:


shap.summary_plot(shap_values, X,plot_type="bar",max_display=50) 

