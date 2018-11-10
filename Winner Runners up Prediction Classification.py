# load the important libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from astropy.table import Table
import warnings

# filter simple warning
warnings.simplefilter("ignore")


# load the dataset
data = pd.read_csv('dataset.csv')

#check data from which leagues have been used
print(data['League'].unique())


# print the size of the data
print(data.shape)

# see the columns
print(data.columns)

#print first 10 entries
print(data.head(10))


# create dummy values for Outcome column
Offshoot = pd.get_dummies(data['Outcome'])

# inclusion of the new column to data
data = pd.concat([data,Offshoot],axis=1)


#print top ten entries
print(data.head(10))


#drop unimportant columns
data.drop(['YearFrom','YearTo','Club','Country','League','Outcome', 'Runners Up'],axis=1,inplace=True)

#print top ten entries
print(data.head(10))


# differentiating feature columns and target column
x = data[['Pld','W','D','L','GF','GA' , 'GD' , 'Points']]
y = data['Winner']

x=np.array(x)
y=np.array(y)

# feature scaling on attribute columns
ft_scl = preprocessing.StandardScaler()
ft_scl.fit(x)
ft_scl.transform(x)

kf = KFold(n_splits=5,  shuffle=True)
for train_index, test_index in kf.split(x):

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # we create four lists to score accuracy , precision, recall and f1 scores
    accuracy_set=[]
    precision_set=[]
    recall_set=[]
    f1_score_set=[]

    # we use logistic regression method for classification

    lr = LogisticRegression()
    lr.fit(x_train,y_train)
    prediction = lr.predict(x_test)
    accuracy_set.append(accuracy_score(y_test, prediction))
    precision_set.append(precision_score(y_test, prediction))
    recall_set.append(recall_score(y_test, prediction))
    f1_score_set.append(f1_score(y_test, prediction))

    # Apply sgd classification method for classification

    sgd = SGDClassifier()
    sgd.fit(x_train,y_train)
    prediction = sgd.predict(x_test)
    accuracy_set.append(accuracy_score(y_test, prediction))
    precision_set.append(precision_score(y_test, prediction))
    recall_set.append(recall_score(y_test, prediction))
    f1_score_set.append(f1_score(y_test, prediction))


    # Apply k nearest neighbours classification method for classification

    knc = KNeighborsClassifier()
    knc.fit(x_train,y_train)
    prediction = knc.predict(x_test)
    accuracy_set.append(accuracy_score(y_test, prediction))
    precision_set.append(precision_score(y_test, prediction))
    recall_set.append(recall_score(y_test, prediction))
    f1_score_set.append(f1_score(y_test, prediction))


    # Apply decision tree classification method for classification

    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    prediction = dtc.predict(x_test)
    accuracy_set.append(accuracy_score(y_test, prediction))
    precision_set.append(precision_score(y_test, prediction))
    recall_set.append(recall_score(y_test, prediction))
    f1_score_set.append(f1_score(y_test, prediction))


    # Apply ada boost classification method for classification

    adc = AdaBoostClassifier()
    adc.fit(x_train, y_train)
    prediction = adc.predict(x_test)
    accuracy_set.append(accuracy_score(y_test, prediction))
    precision_set.append(precision_score(y_test, prediction))
    recall_set.append(recall_score(y_test, prediction))
    f1_score_set.append(f1_score(y_test, prediction))

    # Apply random forest classification method for classification

    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    prediction = rfc.predict(x_test)
    accuracy_set.append(accuracy_score(y_test, prediction))
    precision_set.append(precision_score(y_test, prediction))
    recall_set.append(recall_score(y_test, prediction))
    f1_score_set.append(f1_score(y_test, prediction))

    # we create a table to show the accuracy, precision, recall and f1 score of the classification methods used
    t = Table()
    t['classification'] = ['Logistic Regression','SGD ','KNeighbours ','Decision Tree ','AdaBoost ', 'Random Forest']
    t['accuracy'] = accuracy_set
    t['precision'] = precision_set
    t['recall'] = recall_set
    t['f1 score'] = f1_score_set
    print(t)


