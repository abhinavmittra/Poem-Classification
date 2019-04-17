import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from xgboost import XGBClassifier

df = pd.read_csv('dataset.csv')

#Count Vectorizer of HashingVectorizer can also be used here.
vectorizer = TfidfVectorizer()
X=vectorizer.fit_transform(df['content'])
y=vectorizer.fit_transform(df['type'])

print("There are 4 classes for prediction \n")
print(vectorizer.vocabulary_)

names=['Love','Mythology','Nature','Religion']
print("\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

#Processing for fitting to xgBoost
y_train=y_train.nonzero()[1]
y_test=y_test.nonzero()[1]

model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
xg_predictions = [round(value) for value in y_pred]
print("Accuracy of XGBOOST: ",accuracy_score(y_test, xg_predictions)*100)
print(classification_report(y_test,xg_predictions,labels=None,target_names=names))
