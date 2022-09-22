#!/usr/bin/env python
# coding: utf-8

# ## Importing the required Libraries

# In[1]:


import numpy as np     
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.rcParams['figure.figsize']=(15,10)
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_excel('T:\Masters In Data Science\Machine Learning\Projects\Health_Care.xlsx')    ## Import the dataset


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()  ## Null values check


# In[6]:


df.size


# In[7]:


df.shape


# In[8]:


df.drop_duplicates()  ## Drop the duplicate values if there are any this will help to avoid misleading results


# In[9]:


df.shape


# In[10]:


df.describe()


# In[11]:


df.hist()  ## Plot the Histogram to check the distribution of the data


# #### Looks like chol colun has outliers we will remove them in the further steps

# In[14]:


def classs (x):              ## This function will divide data with age-wise groups
    if 0<x<=25:
        return 'Below_25'
    elif 26<x<=35:
        return '25_to_35'
    elif 36<x<=45:
        return '35_to_45'
    elif 46<x<=55:
        return '45_to_55'
    elif 56<x<=65:
        return '55_to_65'
    else:
        return 'Above_65'   


# In[13]:


df['Class'] = df['age'].apply(classs)


# In[15]:


df.head()


# In[16]:


age = df.groupby(['target','Class']).size().reset_index().rename(columns={0:'Count'})


# In[17]:


age


# In[18]:


plt.pie(age['Count'][5:],labels = age['Class'][5:],autopct='%1.2f%%')


# #### Pie chart clearly shows that out of all people having CVD 30.96% people are from 45 to 55 age group

# In[19]:


plt.pie(age['Count'][:5],labels = age['Class'][:5],autopct='%1.2f%%')


# #### Pie chart shows that out of people not having CVD 48.55% people are from 55 to 65 age group

# In[20]:


df[df['target']==1]['age'].hist(alpha = 0.5, color ='b',bins=6,label = 'target = 1',edgecolor='White')
df[df['target']==0]['age'].hist(alpha = 0.5, color ='k',bins=6,label = 'target = 0',edgecolor='White')


# In[21]:


sex = df.groupby(['sex','Class']).size().reset_index().rename(columns={0:'Count'})


# In[22]:


sex


# In[23]:


plt.pie(sex['Count'][0:5],labels = age['Class'][0:5],autopct='%1.2f%%')


# #### Pie chart shows that out of all females 37.50% females are from age group of 55 to 65 years

# In[24]:


plt.pie(sex['Count'][5:],labels = age['Class'][5:],autopct='%1.2f%%')


# #### Pie chart shows that out of all males 34.30% males are from age group of 55 to 65 years

# In[25]:


sns.heatmap(df.corr(), annot=True)  ## Checking the correlation between the variables


# In[26]:


sns.pairplot(df)


# In[27]:


df.columns


# In[28]:


df[['trestbps','target']].corr()


# In[29]:


df.head()


# In[30]:


plt.figure(figsize=(8,6))
sns.boxplot(data=df,x=df['chol'])


# #### There are outliers in the column

# In[31]:


Q1 = df['chol'].quantile(0.25)
Q3 = df['chol'].quantile(0.75)


# In[32]:


IQR = Q3-Q1
IQR


# In[33]:


lower_whisker = Q1-1.5*IQR
upper_whisker = Q3+1.5*IQR


# In[34]:


lower_whisker


# In[35]:


upper_whisker


# In[36]:


new_df = df[df['chol']<upper_whisker]


# In[37]:


plt.figure(figsize=(8,6))
sns.boxplot(data=new_df,x=new_df['chol'])


# #### Let us check by using chi2 test if Blood Pressure of person can detect heart attack
# ##### H0 = Anomalies in trestbps can detect heart attack
# ##### H1 = Anomalies in trestbps can not detect heart attack

# In[38]:


new_df_crosstab_trestbps = pd.crosstab(new_df['trestbps'],new_df['target'],margins=True)
new_df_crosstab_trestbps


# In[39]:


from scipy.stats import chi2_contingency
chi2_contingency(new_df_crosstab_trestbps)


# ##### Here p-value is 0.99 which is more than our 0.05 or acceptance level hence we fail to reject null hypothesis 
# 
# ##### So to conclude Anomalies in trestbps can detect heart attack

# In[40]:


new_df.columns


# ##### H0 = thalassemia is alone major cause of CVD
# ##### H1 = thalassemia is alone not a major cause of CVD

# In[41]:


new_df_crosstab_thal = pd.crosstab(new_df['thal'],new_df['target'],margins=True)
new_df_crosstab_thal


# In[42]:


chi2_contingency(new_df_crosstab_thal)


# ##### Here we can see p-value is very less than 0.05 hence we reject null hypothesis and conclude that 
# ##### thalassemia is alone not a major cause of CVD

# ## Similarly checking for all the variables indivisually how much they can predict the CVD alone

# In[43]:


new_df_crosstab_age = pd.crosstab(new_df['age'],new_df['target'],margins=True)
new_df_crosstab_age.head()


# In[44]:


chi2_contingency(new_df_crosstab_age)


# In[45]:


new_df_crosstab_sex = pd.crosstab(new_df['sex'],new_df['target'])
new_df_crosstab_sex.head()


# In[46]:


chi2_contingency(new_df_crosstab_sex)


# In[47]:


new_df_crosstab_cp = pd.crosstab(new_df['cp'],new_df['target'])
new_df_crosstab_cp.head()


# In[48]:


chi2_contingency(new_df_crosstab_cp)


# In[49]:


new_df_crosstab_chol = pd.crosstab(new_df['chol'],new_df['target'])
new_df_crosstab_chol.head()


# In[50]:


chi2_contingency(new_df_crosstab_chol)


# In[51]:


new_df_crosstab_fbs = pd.crosstab(new_df['fbs'],new_df['target'])
chi2_contingency(new_df_crosstab_fbs)


# In[52]:


new_df_crosstab_restecg = pd.crosstab(new_df['restecg'],new_df['target'])
chi2_contingency(new_df_crosstab_restecg)


# In[53]:


new_df_crosstab_thalach = pd.crosstab(new_df['thalach'],new_df['target'])
chi2_contingency(new_df_crosstab_thalach)


# In[54]:


new_df_crosstab_exang = pd.crosstab(new_df['exang'],new_df['target'])
chi2_contingency(new_df_crosstab_exang)


# In[55]:


new_df_crosstab_oldpeak = pd.crosstab(new_df['oldpeak'],new_df['target'])
chi2_contingency(new_df_crosstab_oldpeak)


# In[56]:


new_df_crosstab_slope = pd.crosstab(new_df['slope'],new_df['target'])
chi2_contingency(new_df_crosstab_slope)


# In[57]:


new_df_crosstab_ca = pd.crosstab(new_df['ca'],new_df['target'])
chi2_contingency(new_df_crosstab_ca)


# ## Now let us build a Logistic regression model using all the variables except class as it was only prepared for getting insights

# In[58]:


features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']
x = new_df[features]
y = new_df['target']


# In[59]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33)


# In[60]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[61]:


logreg.fit(x_train,y_train)


# In[62]:


y_pred = logreg.predict(x_test)


# In[63]:


from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score


# In[64]:


a = accuracy_score(y_test,y_pred)
a


# In[65]:


p = precision_score(y_test,y_pred)
p


# In[66]:


r = recall_score(y_test,y_pred)
r


# In[67]:


f = f1_score(y_test,y_pred)
f


# In[68]:


from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test,y_pred)
cm1


# In[69]:


from sklearn.metrics import ConfusionMatrixDisplay
cd = ConfusionMatrixDisplay


# In[70]:


conf_matrix1 = cd(confusion_matrix=cm1,display_labels=[False,True])
conf_matrix1.plot()


# #### We have Built a very accurate model with accuracy score of 86.87% 

# ## Now we will build a random forest model to see if we can achieve more accuracy than LOGISTIC REGRESSION Model

# In[71]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=800)


# In[72]:


rf.fit(x_train,y_train)


# In[73]:


y_pred1 = rf.predict(x_test)


# In[74]:


a2 = accuracy_score(y_test,y_pred1)
a2


# In[75]:


p2 = precision_score(y_test,y_pred1)
p2


# In[76]:


r2 = recall_score(y_test,y_pred1)
r2


# In[77]:


f2 = f1_score(y_test,y_pred1)
f2


# In[78]:


cm2 = confusion_matrix(y_test,y_pred1)
cm2


# In[79]:


conf_matrix2 = cd(confusion_matrix=cm2,display_labels=[False,True])
conf_matrix2.plot()


# #### We are getting somewhat similar accuracy for both the models that is around 84% to 86% 
# ## We have checked whether indivisual parameters affect the CVD
# ## Based on p-values obtained from the previous tests we will select only important features and check if we can improve the accuracy of our model

# In[80]:


new_df.columns


# In[81]:


imp_features = ['age','trestbps','chol','fbs','thalach','restecg','oldpeak']
x1 = new_df[imp_features]
y1 = new_df['target']


# In[82]:


x1_train,x1_test,y1_train,y1_test = train_test_split(x1,y1,test_size=0.33)


# In[83]:


rf.fit(x1_train,y1_train)


# In[84]:


y_pred2 = rf.predict(x1_test)


# In[85]:


a3 = accuracy_score(y1_test,y_pred2)
a3


# In[86]:


p3 = precision_score(y1_test,y_pred2)
p3


# In[87]:


r3 = recall_score(y1_test,y_pred2)
r3


# In[88]:


f3 = f1_score(y1_test,y_pred2)
f3


# In[89]:


cm3 = confusion_matrix(y1_test,y_pred2)
cm3


# In[90]:


conf_matrix3 = cd(confusion_matrix=cm3,display_labels=[False,True])
conf_matrix3.plot()


# ## We can clearly see the drop in accuracy which shows that more features required for predicting CVD presence for a patient hence we will continue with previous models

# # Thank You
