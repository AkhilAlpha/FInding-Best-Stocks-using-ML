# Import necessary libraries
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,roc_auc_score
# Load the dataset

stock_data = pd.read_csv(r'C:\Users\Akhil\Desktop\final project\2018_US_Stock_Data.csv')

unwanted_cols = []
# Finding the coloumns where variance is zero
for val in stock_data.var().iteritems():
    if val[1]==0:
        unwanted_cols.append(val[0])
#print("Cols with variance zero" ,unwanted_cols)
## 'Unnamed: 0' and '2019 PRICE VAR [%]' are unwanted coloumns
unwanted_cols.extend(['2019 PRICE VAR [%]'])
stock_data.drop(columns =unwanted_cols, inplace=True, axis=1)
#print("Total Removed columns: ", unwanted_cols)

for col in stock_data.columns:
    if stock_data[col].isnull().sum():
        if stock_data[col].dtype in ['float64','int64']:
            stock_data[col].fillna(value=stock_data[col].median(), inplace=True)
        elif stock_data[col].dtype == 'object':
            stock_data[col].fillna(value=stock_data[col].mode(), inplace=True)

y = stock_data['Class'] #target variable
x =  stock_data.drop(['Unnamed: 0','Class'], axis=1)



cat_cols = []
for col in x.columns:
    if x[col].dtype not in ['float64','int64']:
        cat_cols.append(col)
#print("The categorical cols in the dataset are: " ,cat_cols)

x_1 = pd.get_dummies(x, columns=cat_cols, drop_first=True)

#Feature Selection
# Adding constant

import statsmodels.api as sm
x_2=sm.add_constant(x_1)

from sklearn.linear_model import LogisticRegression

x_train_2,x_test_2,y_train_2,y_test_2=train_test_split(
     x_2,y,test_size=0.2,random_state=10)

x_train_2.shape,x_test_2.shape,y_train_2.shape,y_test_2.shape

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

col_to_be_removed = ger_high_correlated_cols(x_2, 0.70)

features_2 = list(set(x_2.columns)-col_to_be_removed)

x_3 = x_2[features_2]

x_train_3,x_test_3,y_train_3,y_test_3=train_test_split(
     x_3,y,test_size=0.2,random_state=10)

x_train_3.shape,x_test_3.shape,y_train_3.shape,y_test_3.shape
logR_3=sm.Logit(y_train_3,x_train_3)

# Fit the model
logR_3=logR_3.fit()

summary_result = logR_3.summary2().tables[1]
significant_cols = list(summary_result[summary_result['P>|z|'] <0.05].index.values)

x_4 = x_3[significant_cols]

x_train_4,x_test_4,y_train_4,y_test_4=train_test_split(
     x_4,y,test_size=0.2,random_state=10)

x_train_4.shape,x_test_4.shape,y_train_4.shape,y_test_4.shape

logR_4=sm.Logit(y_train_4,x_train_4)

# Fit the model
logR_4=logR_4.fit()

summart_result = logR_4.summary2().tables[1]
significant_cols = list(summart_result[summart_result['P>|z|'] <0.05].index.values)

## Cols with p value lesser than 0.05 are selected as significant cols


x_5 = x_4[significant_cols]


x_train_5,x_test_5,y_train_5,y_test_5=train_test_split(
     x_5,y,test_size=0.2,random_state=10)

x_train_5.shape,x_test_5.shape,y_train_5.shape,y_test_5.shape

logR_5=sm.Logit(y_train_5,x_train_5)

# Fit the model
logR_5=logR_5.fit()


summart_result = logR_5.summary2().tables[1]
significant_cols = list(summart_result[summart_result['P>|z|'] <0.05].index.values)

## Cols with p value lesser than 0.05 are selected as significant cols


x_6 = x_5[significant_cols]



x_train_6,x_test_6,y_train_6,y_test_6=train_test_split(
     x_6,y,test_size=0.2,random_state=10)

x_train_6.shape,x_test_6.shape,y_train_6.shape,y_test_6.shape

logR_6=sm.Logit(y_train_6,x_train_6)

# Fit the model
logR_6=logR_6.fit()


logR_6.params

#Insight:Therse are the 12 most important parameters in determination of target variables



insignificant_col={'interestCoverage','Operating Income','Capital Expenditure'}

signi_col=list(set(x_6.columns)-insignificant_col)

x_7=x_6[signi_col]


x_7['Class']=y
x_7['Name']=stock_data['Unnamed: 0']

stock_new=x_7

stock_yes=stock_new[stock_new['Class']==1]


stock_no=stock_new[stock_new['Class']==0]


from sklearn.utils import resample

stock_no_up=resample(stock_no, replace=True,random_state=100,n_samples=2000)


stock_yes_down=resample(stock_yes, replace=False,random_state=100,n_samples=2500)

### Creating the dataset by combining

stock_new=pd.concat([stock_no_up,stock_yes_down])


from sklearn.utils import shuffle
stock_new=shuffle(stock_new)



Y = stock_new['Class'] #target variable
X = stock_new.drop(['Class','Name'], axis=1)




from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

x_scaled=scaler.fit_transform(X)

x_train_6,x_test_6,y_train_6,y_test_6=train_test_split(
     x_scaled,Y,test_size=0.2,random_state=10)

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(random_state=100)

rfc.fit(x_train_6,y_train_6)


cm=confusion_matrix(y_test_6,rfc.predict(x_test_6))
report=classification_report(y_test_6,rfc.predict(x_test_6))
score=roc_auc_score(y_test_6,rfc.predict(x_test_6))


#Extensive Parameter Tuning
from sklearn.model_selection import GridSearchCV

rfc_gs=GridSearchCV(rfc,{'n_estimators':range(75,125),
                        'criterion':['gini','entropy','log_loss']})


rfc_gs.fit(x_train_6,y_train_6)

rfc_gs.best_params_

rfc_new=RandomForestClassifier(n_estimators=83,
                               criterion='gini',random_state=10)

rfc_new.fit(x_train_6,y_train_6)


# Perfromance

cm=confusion_matrix(y_test_6,rfc_new.predict(x_test_6))
report=classification_report(y_test_6,rfc_new.predict(x_test_6))
score=roc_auc_score(y_test_6,rfc_new.predict(x_test_6))

rfc_new.feature_importances_

# Get the feature importances
importances = rfc_new.feature_importances_
features = x_6.columns


# Define the Streamlit app
def app():
    st.title("Stock Predictor")
    st.write("Use this app to predict whether to buy, sell, or hold a stock based on its financial indicators.")

    # Display the feature importances
    st.subheader("Feature Importances")
    imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
    imp_df = imp_df.sort_values("Importance", ascending=False)
    st.dataframe(imp_df)

    # Allow the user to input the stock data
    st.subheader("Enter Stock Data")
    data_input = {}
    for feature in features:
        data_input[feature] = st.number_input(f"Enter {feature}")
    data_input = pd.DataFrame(data_input, index=[0])

    # Predict the action and display the result
    prediction = clf.predict(data_input)
    st.subheader("Prediction")
    st.write(prediction[0])


# Run the app
if __name__ == '__main__':
    app()
