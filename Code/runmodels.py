import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import torch as T
import numpy 
import csv

#neural network model of the classification
biasdata = pd.read_csv(os.path.join('mediabiastrn.tsv'), header=0, sep='\t', index_col=False)
X_train = biasdata.text
y_train = biasdata.label


#DataLoader(test_iter, , shuffle=shuffle)
testdata = pd.read_csv(os.path.join('mediabiastst.tsv'), header=0, sep='\t', index_col=False, engine='python')    

ooddata = pd.read_csv(os.path.join('MBICtest.tsv'), header=0, sep='\t', index_col=False, engine='python')
valdata = pd.read_csv(os.path.join('mediabiasval.tsv'), header=0, sep='\t', index_col=False, engine='python')
y_val = valdata['label'].tolist()
X_val = valdata['text'].tolist()
X_ood = ooddata['text'].tolist()
y_ood = ooddata['label'].tolist()
X_test = testdata['text'].tolist()
y_test = testdata['label'].tolist()

model = RandomForestClassifier(n_estimators=60)

vectorizer = TfidfVectorizer(lowercase=False, decode_error='replace')


train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
val_vectors = vectorizer.transform(X_val)
ood_vectors = vectorizer.transform(X_ood)
#print(test_vectors)

model.fit(train_vectors, y_train)
#model.transform(test_vectors, y_test)
pred = model.predict(test_vectors)
from sklearn.metrics import accuracy_score
predv = model.predict(val_vectors)
predo = model.predict(ood_vectors)
acc = accuracy_score(pred, y_test)
other = accuracy_score(predv, y_val)
third = accuracy_score(predo, y_ood)
loader = zip(pred, X_test)
print("The rf accuracy on the test set: {:.4f}".format(acc))     
print("The rf accuracy on the val set: {:.4f}".format(other))    
print("The rf accuracy on the ood set: {:.4f}".format(third))     