import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import classification_report,confusion_matrix

# Importing dataset
pd.set_option('display.max_columns',500)
pd.set_option('display.max_rows',500)
dataset = pd.read_csv('toy_dataset.csv')
dataset.describe()
dataset.head()
'''''''''''''''''''''''''''''''''''''''''''''''
#
dataset.hist(column='Illness')
#
dataset['Gender'].value_counts().plot(kind='barh', rot=0, title='Gender Distribution', figsize=(15,3))
#
male_df = dataset[dataset['Gender'] == 'Male']
female_df = dataset[dataset['Gender'] == 'Female']

x = pd.Series(male_df['Income'])
y = pd.Series(female_df['Income'])

plt.figure(figsize=(16,7))

bins = np.linspace(0, 175000, 200)

plt.hist(x, bins, alpha=0.5, label='Male Income')
plt.hist(y, bins, alpha=0.5, label='Female Income')
plt.legend(loc='upper right', prop={'size' : 28})
plt.xlabel('Income')
plt.ylabel('Frequency', rotation=0)
plt.rc('axes', labelsize=10) 
plt.rc('axes', titlesize=30) 
plt.title('Income Distribution by Gender')
plt.show()
#
new_df = dataset[dataset['City'] == 'New York City']
los_df = dataset[dataset['City'] == 'Los Angeles']
bos_df = dataset[dataset['City'] == 'Boston']
moun_df = dataset[dataset['City'] == 'Mountain View']
wash_df = dataset[dataset['City'] == 'Washington']
aus_df = dataset[dataset['City'] == 'Austin']
san_df = dataset[dataset['City'] == 'San Diego']
dal_df = dataset[dataset['City'] == 'Dallas']

a = pd.Series(new_df['Income'])
b = pd.Series(los_df['Income'])
c = pd.Series(bos_df['Income'])
d = pd.Series(moun_df['Income'])
e = pd.Series(wash_df['Income'])
f = pd.Series(aus_df['Income'])
g = pd.Series(san_df['Income'])
h = pd.Series(dal_df['Income'])

plt.figure(figsize=(16,7))

bins = np.linspace(0, 175000, 200)

plt.hist(a, bins, alpha=0.5, label='New York City')
plt.hist(b, bins, alpha=0.5, label='Los Angeles')
plt.hist(c, bins, alpha=0.5, label='Boston', color='cyan')
plt.hist(d, bins, alpha=0.5, label='Mountain View', color='crimson')
plt.hist(e, bins, alpha=0.5, label='Washington D.C.', color='Black')
plt.hist(f, bins, alpha=0.5, label='Austin', color='Gold')
plt.hist(g, bins, alpha=0.5, label='San Diego', color='DarkBlue')
plt.hist(h, bins, alpha=0.5, label='Dallas', color='Lime')
plt.legend(loc='upper right', prop={'size' : 22})
plt.xlabel('Income')
plt.ylabel('Frequency', rotation=0)
plt.rc('axes', labelsize=10) 
plt.rc('axes', titlesize=30) 
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20) 
plt.title('Income Distribution by City')
plt.show()
#
dataset['City'].value_counts().plot(kind='barh', 
                               rot=0, 
                               title='City Counts', 
                               figsize=(15,5))
#
dataset['Illness'].value_counts().plot(kind='barh', 
                                  title='Illness counts', 
                                  figsize=(15,3))
#
dataset['City'] = dataset['City'].astype('category')
dataset['Illness'] = dataset['Illness'].astype('category')
dataset['Gender'] = dataset['Gender'].astype('category')
#
sns.boxplot(data = dataset, x = "City", y = "Income")
sns.boxplot(data = dataset, x = "Gender", y = "Income")
#
g=sns.FacetGrid(dataset,row='City', col='Gender', hue = 'City')
g=g.map(plt.hist, 'Income' )
#
h=sns.FacetGrid(dataset, col = 'City')
h=h.map(sns.boxplot, 'Gender','Income' )
#check skewness
dataset.skew()
#
dataset.plot(kind='box', subplots=True, layout=(4,4), fontsize=8, figsize=(14,14))
plt.show()
'''''''''''''''''''''''''''''''''''''''''''''''''''
#remove duplicates
dataset.drop_duplicates(keep=False,inplace=True)
#count null values
dataset.isnull().sum()
#correlation
import seaborn as sns
corr=dataset.corr()
sns.heatmap(data=corr,square=True,annot=True,cbar=True)

#Target variable converted into binary values
dataset['Illness'] = dataset['Illness'].replace(['No'], 0)
dataset['Illness'] = dataset['Illness'].replace(['Yes'], 1)

#outliers detections
outliers=[]
def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

detect_outlier(dataset['Income'])

#Dropping Number column since it is an index only
del dataset['Number']
#Removal of negative income
dataset = dataset.drop([245], axis=0)

#vif cal
vif_df = dataset._get_numeric_data() #drop non-numeric cols

def vif_cal(input_data, dependent_col):
    import statsmodels.formula.api as sm
    x_vars = input_data.drop([dependent_col],axis=1)
    xvar_names = x_vars.columns
    for i in range(0,len(xvar_names)):
        y = x_vars[xvar_names[i]]
        x = x_vars[xvar_names.drop(xvar_names[i])]
        rsq = sm.ols("y~x",x_vars).fit().rsquared
        vif = round(1/(1-rsq),2)
        print(xvar_names[i], "VIF: ", vif)
        
vif_cal(vif_df, 'Illness')
dataset.head(5)
#one hot encoding:to convert categorical data to numbers
dataset=pd.get_dummies(dataset,columns=['City','Gender'])
#train test split
X =dataset.iloc[:,[0,1,3,4,5,6,7,8,9,10,11,12]]
y=dataset.iloc[:,2]
#train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#########################
#Logistic Regression 
classifier = LogisticRegression(random_state=1)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test) #To predict Y values
#metrics
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
#########################
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#metrics
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
############################################
#random_forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
#metrics
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)



############################################
#svm
from sklearn import svm
model = svm.LinearSVC(loss='squared_hinge', dual=False) 
model.fit(X, y)
model.score(X, y)
y_pred= model.predict(X_test)
####################################
#KNN
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=4)  
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
##########################
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
###########################################
#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
#get accuracy
print('Accuracy of decision tree classifier: {:.2f}'.format(classifier.score(X_test, y_test)))
#performance metrics
from sklearn.metrics import precision_recall_fscore_support
all=precision_recall_fscore_support(y_test, y_pred, average='micro')
print('Precision score=',all[0]*100)
print('Recall score=',all[1]*100)
print('F1 score=',all[2]*100)

