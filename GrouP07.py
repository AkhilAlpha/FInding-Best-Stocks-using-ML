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
unwanted_cols.extend(['Unnamed: 0','2019 PRICE VAR [%]'])
stock_data.drop(columns =unwanted_cols, inplace=True, axis=1)
print("Total Removed columns: ", unwanted_cols)


# In[5]:


for col in stock_data.columns:
    if stock_data[col].isnull().sum():
        if stock_data[col].dtype in ['float64','int64']:
            stock_data[col].fillna(value=stock_data[col].median(), inplace=True)
        elif stock_data[col].dtype == 'object':
            stock_data[col].fillna(value=stock_data[col].mode(), inplace=True)


# In[6]:


stock_data.info()


# In[7]:


stock_data.describe()


# In[8]:


y = stock_data['Class'] #target variable
x =  stock_data.drop(['Class'], axis=1)


# In[9]:


y


# In[10]:


x


# In[11]:


cat_cols = []
for col in x.columns:
    if x[col].dtype not in ['float64','int64']:
        cat_cols.append(col)
print("The categorical cols in the dataset are: " ,cat_cols)


# In[12]:


x[cat_cols[0]].unique()


# In[13]:


x_1 = pd.get_dummies(x, columns=cat_cols, drop_first=True) 
x_1


# In[14]:


y


# # Logistic Regression

# In[15]:


# Adding constant

import statsmodels.api as sm
x_2=sm.add_constant(x_1)
x_2.shape


# In[16]:


from sklearn.linear_model import LogisticRegression


# In[17]:


x_train_2,x_test_2,y_train_2,y_test_2=train_test_split(
     x_2,y,test_size=0.2,random_state=10)

x_train_2.shape,x_test_2.shape,y_train_2.shape,y_test_2.shape


# In[18]:


logR_2=sm.Logit(y_train_2,x_train_2)

#Fit the model
logR_2=logR_2.fit()


# Columns are highly correlated therefore it is showing Singular matrix error

# ## Highly Correlated columns

# In[19]:


def ger_high_correlated_cols(dataset, threshold):
    col_corr = set() # list of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[20]:


x_2


# In[21]:


col_to_be_removed = ger_high_correlated_cols(x_2, 0.70)
print("No of cols to be removed: ", len(col_to_be_removed))
col_to_be_removed


# In[22]:


features_2 = list(set(x_2.columns)-col_to_be_removed)


# In[23]:


x_3 = x_2[features_2]


# In[24]:


x_3


# In[25]:


x_train_3,x_test_3,y_train_3,y_test_3=train_test_split(
     x_3,y,test_size=0.2,random_state=10)

x_train_3.shape,x_test_3.shape,y_train_3.shape,y_test_3.shape


# In[26]:


logR_3=sm.Logit(y_train_3,x_train_3)

# Fit the model
logR_3=logR_3.fit()


# In[27]:


logR_3.summary2()


# In[28]:


summart_result = logR_3.summary2().tables[1]
significant_cols = list(summart_result[summart_result['P>|z|'] <0.05].index.values)
print(len(significant_cols))
print(significant_cols)
## Cols with p value lesser than 0.05 are selected as significant cols 


# In[29]:


x_4 = x_3[significant_cols]


# In[30]:


x_4


# In[31]:


x_train_4,x_test_4,y_train_4,y_test_4=train_test_split(
     x_4,y,test_size=0.2,random_state=10)

x_train_4.shape,x_test_4.shape,y_train_4.shape,y_test_4.shape


# In[32]:


logR_4=sm.Logit(y_train_4,x_train_4)

# Fit the model
logR_4=logR_4.fit()


# In[33]:


logR_4.summary2()


# In[34]:


summart_result = logR_4.summary2().tables[1]
significant_cols = list(summart_result[summart_result['P>|z|'] <0.05].index.values)
print(len(significant_cols))
print(significant_cols)
## Cols with p value lesser than 0.05 are selected as significant cols


# In[35]:


x_5 = x_4[significant_cols]


# In[36]:


x_5


# In[37]:


x_train_5,x_test_5,y_train_5,y_test_5=train_test_split(
     x_5,y,test_size=0.2,random_state=10)

x_train_5.shape,x_test_5.shape,y_train_5.shape,y_test_5.shape


# In[38]:


logR_5=sm.Logit(y_train_5,x_train_5)

# Fit the model
logR_5=logR_5.fit()


# In[39]:


logR_5.summary2()


# In[40]:


summart_result = logR_5.summary2().tables[1]
significant_cols = list(summart_result[summart_result['P>|z|'] <0.05].index.values)
print(len(significant_cols))
print(significant_cols)
## Cols with p value lesser than 0.05 are selected as significant cols


# In[41]:


x_6 = x_5[significant_cols]
x_6


# In[42]:


x_train_6,x_test_6,y_train_6,y_test_6=train_test_split(
     x_6,y,test_size=0.2,random_state=10)

x_train_6.shape,x_test_6.shape,y_train_6.shape,y_test_6.shape


# In[43]:


logR_6=sm.Logit(y_train_6,x_train_6)

# Fit the model
logR_6=logR_6.fit()


# In[44]:


logR_6.summary2()


# In[45]:


logR_6.params


# Insight:
#        Therse are the 12 most important parameters in determination of target variables

# In[46]:


y_pred_6=logR_6.predict(x_test_6)
y_pred_6


# In[47]:


pred_df_6=pd.DataFrame({'Actual Class':y_test_6,
                      'Predicted Prob.':y_pred_6})
pred_df_6


# In[48]:


# Adding a col to the DF

pred_df_6['Predicted Class']=pred_df_6['Predicted Prob.'].map(
    lambda x: 1 if x >0.5 else 0 )


# In[49]:


pred_df_6


# In[50]:


pred_df_6['Actual Class'].iloc[1]


# In[51]:


len(pred_df_6)


# In[52]:


count=0
for val in range(len(pred_df_6)):
    if pred_df_6['Actual Class'].iloc[val]!=pred_df_6['Predicted Class'].iloc[val]:
        count=count+1
print('The no of misclassification is',count)


# In[53]:


222/859


# Insight:
# 
#     There is around 26% of misclassification. So, Logistic Regression Model is only 74% accurate.

# In[54]:


cm=confusion_matrix(pred_df_6['Actual Class'],pred_df_6['Predicted Class'])
report=classification_report(pred_df_6['Actual Class'],pred_df_6['Predicted Class'])
score=roc_auc_score(pred_df_6['Actual Class'],pred_df_6['Predicted Class'])

sns.heatmap(cm,annot=True)
print(' The Report:\n', report)
print(' The ROC-AUC-Score:',score)


# ## Hyperparameter Tuning

# In[55]:


from sklearn.model_selection import GridSearchCV


# In[56]:


from sklearn.linear_model import LogisticRegression


# In[57]:


grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(x_train_6,y_train_6)


# In[58]:


logreg_cv.best_params_


# In[59]:


logreg_cv.best_score_


# In[60]:


logR_7=LogisticRegression()
logR_7.fit(x_train_6,y_train_6)


# In[61]:


score = logR_7.score(x_test_6, y_test_6)
print(score)


# ## Important features

# In[62]:


x_6


# In[63]:


insignificant_col={'interestCoverage','Operating Income','Capital Expenditure'}


# In[64]:


signi_col=list(set(x_6.columns)-insignificant_col)


# In[65]:


x_7=x_6[signi_col]


# In[66]:


x_7


# In[67]:


y


# In[68]:


type(y)


# In[69]:


type(x)


# In[70]:


x_7['Class']=y


# In[71]:


stock_new=x_7


# In[72]:


stock_new


# ## Dealing with imbalanced dataset

# In[73]:


stock_new['Class'].value_counts()


# In[74]:


1346/(1346+2945)


# Dataset is quite unbalanced as dataset contains 69% of 1 class and 39% of 0 class

# ## Resolving the Imbalance

# In[75]:


stock_yes=stock_new[stock_new['Class']==1]


# In[76]:


stock_yes.shape


# In[77]:


stock_no=stock_new[stock_new['Class']==0]
stock_no.shape


# ### Upsampling

# In[78]:


from sklearn.utils import resample

stock_no_up=resample(stock_no, replace=True,random_state=100,n_samples=2000)
stock_no_up.shape


# ### Downsampling

# In[79]:


from sklearn.utils import resample

stock_yes_down=resample(stock_yes, replace=False,random_state=100,n_samples=2500)
stock_yes_down.shape


# ### Creating the dataset by combining

# In[80]:


stock_new=pd.concat([stock_no_up,stock_yes_down])
stock_new.shape


# In[81]:


stock_new


# ### Shuffling

# In[82]:


from sklearn.utils import shuffle
stock_new=shuffle(stock_new)
stock_new


# In[83]:


y = stock_new['Class'] #target variable
x = stock_new.drop(['Class'], axis=1)


# ## Standardisation of features

# In[84]:


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

x_scaled=scaler.fit_transform(x)
x_scaled


# In[85]:


x_train_6,x_test_6,y_train_6,y_test_6=train_test_split(
     x_scaled,y,test_size=0.2,random_state=10)

x_train_6.shape,x_test_6.shape,y_train_6.shape,y_test_6.shape


# ## Decision Tree

# In[86]:


from sklearn.tree import DecisionTreeClassifier

dt_1=DecisionTreeClassifier()

## Training the model

dt_1=dt_1.fit(x_train_6,y_train_6)


# In[87]:


dt_1


# In[88]:


y_pred_1=dt_1.predict(x_test_6)
y_pred_1


# In[89]:


cm=confusion_matrix(y_test_6,y_pred_1)
report=classification_report(y_test_6,y_pred_1)
score=roc_auc_score(y_test_6,y_pred_1)

sns.heatmap(cm,annot=True)
print(' The Report:\n', report)
print(' The ROC-AUC-Score:',score)


# In[90]:


from sklearn.tree import plot_tree
plot_tree(dt_1);


# In[91]:


plt.figure(figsize=(15,15))
plot_tree(dt_1,filled=True);


# ## Decision Rules

# In[92]:


from sklearn.tree import export_text

text=export_text(dt_1,feature_names=list(x.columns))
print(text)


# In[93]:


dt_1.criterion


# In[94]:


dt_1.tree_.max_depth


# ## Modifying DT model

# In[95]:


dt_2=DecisionTreeClassifier(criterion='entropy',max_depth=10)

dt_2=dt_2.fit(x_train_6,y_train_6)
y_pred_2=dt_2.predict(x_test_6)


# In[96]:


cm=confusion_matrix(y_test_6,y_pred_2)
report=classification_report(y_test_6,y_pred_2)
score=roc_auc_score(y_test_6,y_pred_2)

sns.heatmap(cm,annot=True)
print(' The Report:\n', report)
print(' The ROC-AUC-Score:',score)


# In[97]:


plt.figure(figsize=(15,15))
plot_tree(dt_2,filled=True);


# ## Support Vector Classifier

# In[98]:


from sklearn.svm import SVC

svc_lin=SVC(kernel='linear',probability=True)
svc_lin=svc_lin.fit(x_train_6, y_train_6)
y_pred=svc_lin.predict(x_test_6)
y_pred


# In[99]:


y_pred_prob=svc_lin.predict_proba(x_test_6)
y_pred_prob


# ### Performance Checking

# In[100]:


from sklearn.metrics import confusion_matrix,classification_report,roc_curve,roc_auc_score

cm=confusion_matrix(y_test_6,y_pred)
report=classification_report(y_test_6,y_pred)
score=roc_auc_score(y_test_6,y_pred)
fpr,tpr,_=roc_curve(y_test_6,y_pred_prob[:,1])

sns.heatmap(cm,annot=True)
print(' The Report:\n', report)
print(' The ROC-AUC-Score:',score)


# In[101]:


plt.plot(fpr,tpr);


# ## Hyper Parameter Tuning

# ### Kernel

# In[102]:


def SVC_tuning_kernel(kernel):
    model=SVC(kernel=kernel)
    model=model.fit(x_train_6,y_train_6)
    y_pred=model.predict(x_test_6)
    cm=confusion_matrix(y_test_6,y_pred)
    report=classification_report(y_test_6,y_pred)
    score=roc_auc_score(y_test_6,y_pred)
    print(' The SVC with kernel:',kernel)
    print()
    print('  ************* ')
    print(' Confusion Matrix:\n',cm)
    print(' The report:\n', report)
    print(' The ROC-AUC-Score:', score)
    sns.heatmap(cm,annot=True);


# In[103]:


# Calling the fn

SVC_tuning_kernel('linear')


# In[104]:


SVC_tuning_kernel('poly')


# In[105]:


SVC_tuning_kernel('rbf')


# In[106]:


SVC_tuning_kernel('sigmoid')


# The best kernel after tuning is 'rbf' or Radial Basis Function Kernel

# ## RBF kernel

# ### Tuning Regularisation Parameter

# In[107]:


def SVC_tuning_C(C_list):
    for c in C_list:
        model=SVC(kernel='rbf',C=c)
        model=model.fit(x_train_6,y_train_6)
        y_pred=model.predict(x_test_6)
        score=roc_auc_score(y_test_6,y_pred)
        print('C:',c,'===>','Score:',score)


# In[108]:


C_list=[0.1,1,2,3,4,5,10,15,20,25,30]


# In[109]:


SVC_tuning_C(C_list)


# In[110]:


C_list=[21,22,23,24,25,26,27,28,29]


# In[111]:


SVC_tuning_C(C_list)


# After tuning the best value for C is 24

# So, the best model is the one with kernel='rbf' and C=0.7

# ### The Final SVC Model

# In[112]:


svc=SVC(kernel='rbf',C=24,probability=True)
svc=svc.fit(x_train_6,y_train_6)
y_pred=svc.predict(x_test_6)
y_pred_prob=svc.predict_proba(x_test_6)

cm=confusion_matrix(y_test_6,y_pred)
score=roc_auc_score(y_test_6,y_pred)
report=classification_report(y_test_6,y_pred)
fpr,tpr,_=roc_curve(y_test_6,y_pred_prob[:,1])

print('The Confusion Matrix:')
sns.heatmap(cm,annot=True)
print('ROC-AUC-Score:',score)
print(' The report:',report)


# # KNN

# In[113]:


from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()

knn.fit(x_train_6, y_train_6)


# ### Model performance

# In[114]:


cm=confusion_matrix(y_test_6,knn.predict(x_test_6))
report=classification_report(y_test_6,knn.predict(x_test_6))
score=roc_auc_score(y_test_6,knn.predict(x_test_6))
print('CM:\n',cm)
print('Report:\n',report)
print('ROC-AUC Curve',score)


# ### Hyper parameter tuning using GridSearchCV

# In[115]:


from sklearn.model_selection import GridSearchCV

knn_gs=GridSearchCV(knn,{'n_neighbors':range(3,25)})

knn_gs.fit(x_train_6,y_train_6)


# In[116]:


knn_gs.best_params_


# ### Best KNN Model

# In[117]:


knn_best=KNeighborsClassifier(n_neighbors=3)
knn_best.fit(x_train_6,y_train_6)


# In[118]:


cm=confusion_matrix(y_test_6,knn_best.predict(x_test_6))
report=classification_report(y_test_6,knn_best.predict(x_test_6))
score=roc_auc_score(y_test_6,knn_best.predict(x_test_6))
print('CM:\n',cm)
print('Report:\n',report)
print('ROC-AUC Curve',score)


# # Ensemblem Learning

# In[119]:


from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(random_state=100)

rfc.fit(x_train_6,y_train_6)


# In[120]:


cm=confusion_matrix(y_test_6,rfc.predict(x_test_6))
report=classification_report(y_test_6,rfc.predict(x_test_6))
score=roc_auc_score(y_test_6,rfc.predict(x_test_6))

print('CM:\n',cm)
print('Report:\n',report)
print('ROC-AUC Curve',score)


# ## Extensive Hyperparameter tuning

# In[121]:


from sklearn.model_selection import GridSearchCV

rfc_gs=GridSearchCV(rfc,{'n_estimators':range(75,125),
                        'criterion':['gini','entropy','log_loss']})


# In[122]:


rfc_gs.fit(x_train_6,y_train_6)


# In[123]:


rfc_gs.best_params_


# In[124]:


rfc_new=RandomForestClassifier(n_estimators=83, 
                               criterion='gini',random_state=10)


# In[125]:


rfc_new.fit(x_train_6,y_train_6)


# In[126]:


# Perfromance

cm=confusion_matrix(y_test_6,rfc_new.predict(x_test_6))
report=classification_report(y_test_6,rfc_new.predict(x_test_6))
score=roc_auc_score(y_test_6,rfc_new.predict(x_test_6))
print('CM:\n',cm)
print('Report:\n',report)
print('ROC-AUC Curve',score)


# In[127]:


rfc_new.feature_importances_


# In[128]:


# Creating a DF

df=pd.DataFrame({'Feature':x.columns,'Feature Imp':rfc_new.feature_importances_})
df


# In[129]:


df=df.sort_values(['Feature Imp'],ascending=False)
df


# In[130]:


# Adding one column

df['Feature Imp Cum']=df['Feature Imp'].cumsum()
df


# In[131]:


sns.barplot(x=df['Feature Imp'],y=df['Feature'],data=df);


# ## Boosting

#  ### Adaboost

# In[132]:


from sklearn.ensemble import AdaBoostClassifier
abc=AdaBoostClassifier(random_state=10)
abc.fit(x_train_6,y_train_6)


# In[133]:


cm=confusion_matrix(y_test_6,abc.predict(x_test_6))
report=classification_report(y_test_6,abc.predict(x_test_6))
score=roc_auc_score(y_test_6,abc.predict(x_test_6))
print('CM:\n',cm)
print('Report:\n',report)
print('ROC-AUC Curve',score)


# In[134]:


abc_gs=GridSearchCV(abc,{'n_estimators':range(25,75),
                        'learning_rate':[0,0.25,0.5,0.75,1]})


# In[135]:


abc_gs.fit(x_train_6, y_train_6)


# In[136]:


abc_gs.best_params_


# In[137]:


## Building the best model

abc_best=AdaBoostClassifier(learning_rate=0.5,n_estimators=44,random_state=10)
abc_best.fit(x_train_6,y_train_6)


# In[138]:


cm=confusion_matrix(y_test_6,abc_best.predict(x_test_6))
report=classification_report(y_test_6,abc_best.predict(x_test_6))
score=roc_auc_score(y_test_6,abc_best.predict(x_test_6))
print('CM:\n',cm)
print('Report:\n',report)
print('ROC-AUC Curve',score)


# ## Building - Gradient Boosting Classifier

# In[139]:


from sklearn.ensemble import GradientBoostingClassifier


# In[140]:


gbc=GradientBoostingClassifier(random_state=10)
gbc.fit(x_train_6,y_train_6)


# In[141]:


cm=confusion_matrix(y_test_6,gbc.predict(x_test_6))
report=classification_report(y_test_6,gbc.predict(x_test_6))
score=roc_auc_score(y_test_6,gbc.predict(x_test_6))
print('CM:\n',cm)
print('Report:\n',report)
print('ROC-AUC Curve',score)


# In[142]:


# Hyper parameter tuning

gbc_gs=GridSearchCV(gbc,{'n_estimators':range(75,125),
                        'max_depth':range(1,5)})


# In[143]:


gbc_gs.fit(x_train_6,y_train_6)


# In[144]:


gbc_gs.best_params_


# ### Best Gradient Boosting model

# In[145]:


grad_best=GradientBoostingClassifier(max_depth=4,n_estimators=124,random_state=10)
grad_best.fit(x_train_6,y_train_6)


# In[146]:


cm=confusion_matrix(y_test_6,grad_best.predict(x_test_6))
report=classification_report(y_test_6,grad_best.predict(x_test_6))
score=roc_auc_score(y_test_6,grad_best.predict(x_test_6))
print('CM:\n',cm)
print('Report:\n',report)
print('ROC-AUC Curve',score)


# ## XGBoost

# In[147]:


pip install xgboost


# In[148]:


from xgboost import XGBClassifier


# In[149]:


xg=XGBClassifier()
xg.fit(x_train_6,y_train_6)


# In[150]:


cm=confusion_matrix(y_test_6,xg.predict(x_test_6))
report=classification_report(y_test_6,xg.predict(x_test_6))
score=roc_auc_score(y_test_6,xg.predict(x_test_6))
print('CM:\n',cm)
print('Report:\n',report)
print('ROC-AUC Curve',score)


# In[151]:


# hyper parameter tuning

xg_gs=GridSearchCV(xg,{'n_estimators':range(75,125),
                        'max_depth':range(1,5)})


# In[152]:


xg_gs.fit(x_train_6,y_train_6)


# In[153]:


xg_gs.best_params_


# ### Best XGBoost model

# In[154]:


xg_best=XGBClassifier(max_depth=4, n_estimators=108)


# In[155]:


xg_best.fit(x_train_6,y_train_6)


# In[156]:


cm=confusion_matrix(y_test_6,xg_best.predict(x_test_6))
report=classification_report(y_test_6,xg_best.predict(x_test_6))
score=roc_auc_score(y_test_6,xg_best.predict(x_test_6))
print('CM:\n',cm)
print('Report:\n',report)
print('ROC-AUC Curve',score)


# The best model is so far is Gradient Boosting model

# ## Selecting the best boosting model

# In[157]:


grad_best.feature_importances_


# In[158]:


df=pd.DataFrame({'Feature':x.columns, 'Imp':grad_best.feature_importances_})
df


# In[159]:


df1=df.sort_values(['Imp'],ascending=False)
df1


# In[160]:


sns.barplot(x=df1['Imp'],y=df1['Feature'],data=df1);


# # K Means Clustering

# ### Finding K using elbow

# In[161]:


from sklearn.cluster import KMeans


# In[162]:


SSD=[]
for k in range(1,35):
    kmeans=KMeans(n_clusters=k,random_state=10)
    kmeans.fit(x_scaled)
    SSD.append(kmeans.inertia_)
plt.plot(range(1,35),SSD);


# In[163]:


SSD=[]
for k in range(1,35):
    kmeans=KMeans(n_clusters=k,random_state=10)
    kmeans.fit(x_scaled)
    SSD.append(kmeans.inertia_)
plt.plot(range(1,35),SSD);
plt.xlim([5,15]);


# K value is 10 or 11 according to Elbow method

# ## Silhouette Method

# In[164]:


from sklearn.metrics import silhouette_score
SS=[]
for k in range(2,35):
    kmeans=KMeans(n_clusters=k, random_state=10)
    kmeans.fit(x_scaled)
    SS.append(silhouette_score(x_scaled,kmeans.predict(x_scaled)))
plt.plot(range(2,35),SS);


# In[165]:


from sklearn.metrics import silhouette_score
SS=[]
for k in range(2,35):
    kmeans=KMeans(n_clusters=k, random_state=10)
    kmeans.fit(x_scaled)
    SS.append(silhouette_score(x_scaled,kmeans.predict(x_scaled)))
plt.plot(range(2,35),SS);
plt.xlim([10,15]);


# Best K value is=11 from Silhouette score

# ## Building the best Model

# In[166]:


k_final=KMeans(n_clusters=11,random_state=10)
k_final.fit(x_scaled)
clusters=k_final.predict(x_scaled)
clusters


# In[167]:


# The cluster centroids

k_final.cluster_centers_


# In[168]:


cm=confusion_matrix(y_test_6,k_final.predict(x_test_6))
report=classification_report(y_test_6,k_final.predict(x_test_6))
score=roc_auc_score(y_test_6,k_final.predict(x_test_6))
print('CM:\n',cm)
print('Report:\n',report)
print('ROC-AUC Curve',score)


# ## DBSCAN

# In[169]:


from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.5, min_samples=5)
clustering_labels = db.fit_predict(x_scaled)


# In[170]:


from sklearn import metrics

score=metrics.silhouette_score(x_scaled,clustering_labels)


# In[171]:


print(score)


# In[172]:


def db_score(val1,val2):
    db=DBSCAN(eps=val1, min_samples=val2)
    clustering_labels = db.fit_predict(x_scaled)
    score=metrics.silhouette_score(x_scaled,clustering_labels)
    return score


# In[173]:


for i in range(1,10):
    for j in range(1,10):
        print(i/10,j,db_score(i/10,j))


# ### Selecting the best one

# In[174]:


# Final

db_score(0.9,7)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Prediction using KNeighborsClassifier

# In[175]:


x_6_new= db.components_
y=db.labels_[db.core_sample_indices_]


# In[176]:


from sklearn.model_selection import train_test_split

X_train_db, X_test_db,y_train_db,y_test_db=train_test_split(x_6_new,y,test_size=0.2, random_state=10)

X_train_db.shape, X_test_db.shape,y_train_db.shape,y_test_db.shape


# In[177]:


from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=5)
kn.fit(X_train_db,y_train_db)


# In[178]:


cm=confusion_matrix(y_test_db,kn.predict(X_test_db))
report=classification_report(y_test_db,kn.predict(X_test_db))
print('CM:\n',cm)
print('Report:\n',report)

