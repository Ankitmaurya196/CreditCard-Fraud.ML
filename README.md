
   # **Import the liabary**
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

   **To load dataset creditcard.csv**
df=pd.read_csv("creditcard.csv")
df.head()

   # Print the shape of the data(df)
df.shape
df.describe

   # Determine number of fraud cases in dataset
fraud = df[df['Class'] == 1]
valid = df[df['Class'] == 0]
outlierFraction = len(fraud)/float(len(valid))
print(outlierFraction)
print('Fraud Cases: {}'.format(len(df[df['Class'] == 1])))
print('Valid Transactions: {}'.format(len(df[df['Class'] == 0])))

  # Code : Print the amount details for Fraudulent Transaction
print(“Amount details of the fraudulent transaction”)
fraud.Amount.describe()

   # Code : Print the amount details for Normal Transaction
print(“details of valid transaction”)
valid.Amount.describe()

  # Code : Plotting the Correlation Matrix
The correlation matrix graphically gives us an idea of how features correlate with each other and can help us predict
what are the features that are most relevant for the prediction.
  # Correlation matrix
corrmat=df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()

   # dividing the X and the Y from the dataset
X=df.drop(['Class'], axis = 1)
Y=df["Class"]
print(X.shape)
print(Y.shape)
#getting just the values for the sake of processing
xData = X.values
yData = Y.values

   # Train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)

   # Building the Random Forest Classifier (RANDOM FOREST)
from sklearn.ensemble import RandomForestClassifier
#random forest model creation
rfc=RandomForestClassifier()
rfc.fit(X_train,Y_train)
yPred=rfc.predict(X_test)

  # Evaluating the classifier
  # printing every score of the classifier
  # scoring in anything
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
 
n_outliers=len(fraud)
n_errors=(yPred !=Y_test).sum()
print("The model used is Random Forest classifier")
 
acc=accuracy_score(Y_test, yPred)
print("The accuracy is {}".format(acc))
 
prec=precision_score(Y_test, yPred)
print("The precision is {}".format(prec))
 
rec=recall_score(Y_test, yPred)
print("The recall is {}".format(rec))
 
f1=f1_score(Y_test, yPred)
print("The F1-Score is {}".format(f1))
 
MCC=matthews_corrcoef(Y_test, yPred)
print("The Matthews correlation coefficient is{}".format(MCC))

   # printing the confusion matrix
LABELS=['Normal','Fraud']
conf_matrix=confusion_matrix(Y_test, yPred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix,xticklabels=LABELS, yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

Comparison with other algorithms without dealing with the imbalancing of the data.





