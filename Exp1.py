#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,roc_auc_score


# In[2]:


stock_data = pd.read_csv('2018_US_Stock_Data.csv')
stock_data.shape


# In[3]:


sns.histplot(data=stock_data, x="2019 PRICE VAR [%]", kde=True);

## Target variable is normaly distributed


# In[4]:


unwanted_cols = []
# Finding the coloumns where variance is zero
for val in stock_data.var().iteritems():
    if val[1]==0:
        unwanted_cols.append(val[0])
print("Cols with variance zero" ,unwanted_cols)
## 'Unnamed: 0' and '2019 PRICE VAR [%]' are unwanted coloumns 
unwanted_cols.extend(['2019 PRICE VAR [%]'])
stock_data.drop(columns =unwanted_cols, inplace=True, axis=1)
print("Total Removed columns: ", unwanted_cols)


# In[5]:


stock_data.isnull().sum()


# In[6]:


for col in stock_data.columns:
    if stock_data[col].isnull().sum():
        if stock_data[col].dtype in ['float64','int64']:
            stock_data[col].fillna(value=stock_data[col].median(), inplace=True)
        elif stock_data[col].dtype == 'object':
            stock_data[col].fillna(value=stock_data[col].mode(), inplace=True)


# In[7]:


stock_data[col].isnull().sum()


# In[8]:


y = stock_data['Class'] #target variable
x =  stock_data.drop(['Unnamed: 0','Class'], axis=1)


# In[9]:


x.head(5)


# In[10]:


y.head(5)


# In[11]:


cat_cols = []
for col in x.columns:
    if x[col].dtype not in ['float64','int64']:
        cat_cols.append(col)
print("The categorical cols in the dataset are: " ,cat_cols)

x_1 = pd.get_dummies(x, columns=cat_cols, drop_first=True) 
x_1


# In[12]:


#Feature Selection
# Adding constant

import statsmodels.api as sm
x_2=sm.add_constant(x_1)
x_2.shape

from sklearn.linear_model import LogisticRegression

x_train_2,x_test_2,y_train_2,y_test_2=train_test_split(
     x_2,y,test_size=0.2,random_state=10)

x_train_2.shape,x_test_2.shape,y_train_2.shape,y_test_2.shape


# In[13]:


#logR_2=sm.Logit(y_train_2,x_train_2)

#Fit the model
#logR_2=logR_2.fit()


# In[14]:


#There highly correlated therefore it is showing Singular matrix error

#Highly Correlated columns
def ger_high_correlated_cols(dataset, threshold):
    col_corr = set() # list of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
    return col_corr
x_2.head(5)


# In[15]:


col_to_be_removed = ger_high_correlated_cols(x_2, 0.70)
print("No of cols to be removed: ", len(col_to_be_removed))


# In[16]:


features_2 = list(set(x_2.columns)-col_to_be_removed)

x_3 = x_2[features_2]
x_3.head(5)


# In[17]:


#1
x_train_3,x_test_3,y_train_3,y_test_3=train_test_split(
     x_3,y,test_size=0.2,random_state=10)

x_train_3.shape,x_test_3.shape,y_train_3.shape,y_test_3.shape
logR_3=sm.Logit(y_train_3,x_train_3)

# Fit the model
logR_3=logR_3.fit()
logR_3.summary2()


# In[19]:


summary_result = logR_3.summary2().tables[1]
significant_cols = list(summary_result[summary_result['P>|z|'] <0.05].index.values)
print(len(significant_cols))
print(significant_cols)
## Cols with p value lesser than 0.05 are selected as significant cols 


# In[20]:


x_4 = x_3[significant_cols]
x_4.head(5)


# In[21]:


#2
x_train_4,x_test_4,y_train_4,y_test_4=train_test_split(
     x_4,y,test_size=0.2,random_state=10)

x_train_4.shape,x_test_4.shape,y_train_4.shape,y_test_4.shape

logR_4=sm.Logit(y_train_4,x_train_4)

# Fit the model
logR_4=logR_4.fit()

logR_4.summary2()


# In[22]:


summart_result = logR_4.summary2().tables[1]
significant_cols = list(summart_result[summart_result['P>|z|'] <0.05].index.values)
print(len(significant_cols))
print(significant_cols)
## Cols with p value lesser than 0.05 are selected as significant cols


# In[23]:


x_5 = x_4[significant_cols]
x_5.head(5)


# In[24]:


x_train_5,x_test_5,y_train_5,y_test_5=train_test_split(
     x_5,y,test_size=0.2,random_state=10)

x_train_5.shape,x_test_5.shape,y_train_5.shape,y_test_5.shape

logR_5=sm.Logit(y_train_5,x_train_5)

# Fit the model
logR_5=logR_5.fit()

logR_5.summary2()


# In[25]:


summart_result = logR_5.summary2().tables[1]
significant_cols = list(summart_result[summart_result['P>|z|'] <0.05].index.values)
print(len(significant_cols))
print(significant_cols)
## Cols with p value lesser than 0.05 are selected as significant cols


# In[26]:


x_6 = x_5[significant_cols]
x_6.head(5)


# In[27]:


x_train_6,x_test_6,y_train_6,y_test_6=train_test_split(
     x_6,y,test_size=0.2,random_state=10)

x_train_6.shape,x_test_6.shape,y_train_6.shape,y_test_6.shape

logR_6=sm.Logit(y_train_6,x_train_6)

# Fit the model
logR_6=logR_6.fit()

logR_6.summary2()


# In[28]:


logR_6.params

#Insight:Therse are the 12 most important parameters in determination of target variables


# In[29]:


insignificant_col={'interestCoverage','Operating Income','Capital Expenditure'}

signi_col=list(set(x_6.columns)-insignificant_col)

x_7=x_6[signi_col]

x_7


# In[30]:


y.value_counts()


# In[31]:


x_7['Class']=y
x_7['Name']=stock_data['Unnamed: 0']


stock_new=x_7

stock_new


# In[32]:


stock_new['Class'].value_counts()


# In[33]:


1346/(1346+2945)


# In[34]:


stock_yes=stock_new[stock_new['Class']==1]

stock_yes.shape

stock_no=stock_new[stock_new['Class']==0]
stock_no.shape


# In[35]:


from sklearn.utils import resample

stock_no_up=resample(stock_no, replace=True,random_state=100,n_samples=2000)


stock_yes_down=resample(stock_yes, replace=False,random_state=100,n_samples=2500)

stock_yes_down.shape

### Creating the dataset by combining

stock_new=pd.concat([stock_no_up,stock_yes_down])
stock_new.shape


# In[36]:


stock_new


# In[37]:


from sklearn.utils import shuffle
stock_new=shuffle(stock_new)
stock_new


# In[38]:


Y = stock_new['Class'] #target variable
X = stock_new.drop(['Class','Name'], axis=1)


# In[122]:


stock_new.info()


# In[39]:


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

x_scaled=scaler.fit_transform(X)
x_scaled


# In[40]:


x_train_6,x_test_6,y_train_6,y_test_6=train_test_split(
     x_scaled,Y,test_size=0.2,random_state=10)

x_train_6.shape,x_test_6.shape,y_train_6.shape,y_test_6.shape


# In[41]:


from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(random_state=100)

rfc.fit(x_train_6,y_train_6)


# In[42]:


cm=confusion_matrix(y_test_6,rfc.predict(x_test_6))
report=classification_report(y_test_6,rfc.predict(x_test_6))
score=roc_auc_score(y_test_6,rfc.predict(x_test_6))

print('CM:\n',cm)
print('Report:\n',report)
print('ROC-AUC Curve',score)


# In[43]:


#Extensive Parameter Tuning
from sklearn.model_selection import GridSearchCV

rfc_gs=GridSearchCV(rfc,{'n_estimators':range(75,125),
                        'criterion':['gini','entropy','log_loss']})


# In[44]:


rfc_gs.fit(x_train_6,y_train_6)


# In[45]:


rfc_gs.best_params_


# In[46]:


rfc_new=RandomForestClassifier(n_estimators=83, 
                               criterion='gini',random_state=10)


# In[47]:


rfc_new.fit(x_train_6,y_train_6)


# In[48]:


# Perfromance

cm=confusion_matrix(y_test_6,rfc_new.predict(x_test_6))
report=classification_report(y_test_6,rfc_new.predict(x_test_6))
score=roc_auc_score(y_test_6,rfc_new.predict(x_test_6))
print('CM:\n',cm)
print('Report:\n',report)
print('ROC-AUC Curve',score)


# In[49]:


rfc_new.feature_importances_


# In[50]:


# Creating a DF

df=pd.DataFrame({'Feature':X.columns,'Feature Imp':rfc_new.feature_importances_})
df


# In[51]:


df=df.sort_values(['Feature Imp'],ascending=False)
df


# In[52]:


# Adding one column

df['Feature Imp Cum']=df['Feature Imp'].cumsum()
df


# In[74]:


df['Rank'] = df.apply(lambda x: (x['Feature Imp'] * 0.5 + x['Feature Imp'] * 0.3 + x['Feature Imp'] * 0.1+x['Feature Imp'] * 0.1), axis=1)
df.sort_values('Rank', ascending=False, inplace=True)
df


# In[113]:


# sort the data frame by 'Feature Imp' column in descending order
df_sorted = df.sort_values(by='Feature Imp', ascending=False)

# select the top four rows
df_top_four = df_sorted.head(4)

# create a dictionary from the top four rows of the data frame
my_imp = df_top_four.set_index('Feature')['Feature Imp'].to_dict()

# print the dictionary
print(my_imp)


# In[128]:


stock_new


# In[132]:


print('Top 10 stocks to Hold:')
print(stock_new[['Name','Class']].loc[stock_new['Class']==1].head(10))
print('Top 10 stocks to sell:')
#print(stock_new)
print(stock_new[['Name','Class']].loc[stock_new['Class']==0].head(10))
#print(stock_new)


# In[127]:


# Iterate through each row of df
for i, row in stock_new.iterrows():
    # Iterate through each column in my_imp
    for col, threshold in my_imp.items():
        # Check if the value in df for this row and column is above or below threshold
        if row[col] > threshold:
            print(f"{col} in row {i} is above threshold ({threshold})")
        else:
            print(f"{col} in row {i} is below threshold ({threshold})")


# In[ ]:





# In[ ]:




