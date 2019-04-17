import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.svm import SVC
df = pd.read_csv('dataset.csv')

#Count Vectorizer of HashingVectorizer can also be used here.
vectorizer = TfidfVectorizer()
X=vectorizer.fit_transform(df['content'])
y=vectorizer.fit_transform(df['type'])

print("There are 4 classes for prediction \n")
print(vectorizer.vocabulary_)

names=['Love','Mythology','Nature','Religion']
print("\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20)

#Processing for fitting to SVC
y_train=y_train.nonzero()[1]
X_test=X_test.todense()
y_test=y_test.nonzero()[1]
#We use linear kernel because text if often linearly seperable

svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)
print("SVM Accuracy :",accuracy_score(y_test,svm_predictions)*100)
print(classification_report(y_test,svm_predictions,labels=None,target_names=names))




