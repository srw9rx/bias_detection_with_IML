#from srw9rx_cnn import SimpleCNN
import sys, os, random, math, sys
#import torch, spacy
import numpy as np
from torch import nn
from tqdm import trange
from sklearn.pipeline import make_pipeline

from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from torch.nn import functional as F
import argparse
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import torch as T
import random
#from srw9rx_cnn import tokenize_fn, reader, eval
from torchtext.data import Field, ReversibleField, Dataset, TabularDataset, BucketIterator, Iterator
import lime.lime_text
import lime
from copy import deepcopy
import csv

print(dir(lime))

parser = argparse.ArgumentParser()
parser.add_argument("--rpath", default='mediabias')
#arser.add_argument('--test_mode', type=str, default='in_dist', help='test mode')
args = vars(parser.parse_args())
print(args)
path = args['rpath']
seed = 1
outpath = ""

test_mode = 'ood'

if test_mode == 'id':
    outpath = os.path.join('mediabiastst.tsv')
elif test_mode == 'ood':
    outpath = os.path.join('MBICtest.tsv')



#(biasdata)
biasdata = pd.read_csv(os.path.join(path+'trn.tsv'), header=0, sep='\t', index_col=False)
X_train = biasdata.text
y_train = biasdata.label
if test_mode == 'id':
    #DataLoader(test_iter, , shuffle=shuffle)
    testdata = pd.read_csv(os.path.join(path+'tst.tsv'), header=0, sep='\t', index_col=False, engine='python')    
else:
    testdata = pd.read_csv(os.path.join('MBICtest.tsv'), header=0, sep='\t', index_col=False, engine='python')

y_test = testdata['label'].tolist()
X_test = testdata['text'].tolist()


model = RandomForestClassifier(n_estimators=60)

vectorizer = TfidfVectorizer(lowercase=False, decode_error='replace')


train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)


#print(test_vectors)

model.fit(train_vectors, y_train)
#model.transform(test_vectors, y_test)
pred = model.predict(test_vectors)
from sklearn.metrics import accuracy_score

acc = accuracy_score(pred, y_test)
loader = zip(pred, X_test)
print("The rf accuracy on the test set: {:.4f}".format(acc))        

M=50
def sample_shapley(model, pred, idx, text):
  phi = 0
  for _ in range(M):
    # TODO: Obtain the set x_F\{i}
    F_set = deepcopy(text)
    F_set.remove(text[idx])

    # TODO: Sample the set x_Sm from x_F\{i}
    samplelength = random.randint(1, len(F_set)-1)
    Sm_set = random.sample(F_set, samplelength) #randomly sample the words like we did in the previous PA 
    # TODO: Construct the set x_Sm_i - the same 
    Sm_i_set = deepcopy(Sm_set)
    Sm_i_set.append(text[idx]) #removes the value from the sample text 
    # TODO: concatenate the words in x_Sm and x_Sm_i into a text (a string with words seperated by " ") respectively
    Sm_text = ""
    for word in Sm_set:
      Sm_text = Sm_text+word+ " "

    Sm_text = vectorizer.transform([Sm_text])
    Sm_i_text = ""
    for word in Sm_i_set:
      Sm_i_text = Sm_i_text+(word+ " ")
    Sm_i_text = vectorizer.transform([Sm_i_text])

    # TODO: Call the model and obtain the output probabilities on Sm_text and Sm_i_text respectively
    pred_1 = model.predict_proba(Sm_text)[0][pred] #get for the proper class, always 0 because there is only one value in preds
    pred_2 = model.predict_proba(Sm_i_text)[0][pred]
    #print(type(pred_1), pred_2)
    # TODO: Compute the marginal contribution of x_i given x_Sm
    delta_p = pred_2-pred_1

    phi += delta_p

  return phi/M
from sklearn.preprocessing import FunctionTransformer

pipeline = make_pipeline(
     vectorizer, 
     FunctionTransformer(lambda x: x.todense(), accept_sparse=True), 
     model
)
#pipeline = make_pipeline(todense(vectorizer), model)
limeexplainer = lime.lime_text.LimeTextExplainer(kernel_width=100, verbose=False, class_names=['0','1'], feature_selection='auto', bow=False, mask_string=None, random_state=None, char_level=False)

shapleyvals = []
limevals = []
for label, inputs in tqdm(enumerate(loader)):
  pred = inputs[0]
  text = inputs[1].split(" ")
  thisvalshap = []
  thisvallime = []
  for idx in range(len(text)):
    if label%10 == 0:
      thisvalshap.append(sample_shapley(model, pred, idx, text)) #append the word to the list
    else:
      thisvalshap.append('0')    
    thisvallime.append('0')

  #print(thisvalshap, '\n', thisvallime)
  shaps = zip(text, thisvalshap)
  shaps = sorted(shaps, key=lambda x: x[1], reverse=True)
  if len(shaps) > 6:
    shaps = shaps[0::6]
  shapleyvals.append(shaps) #zip together the shapley value and the word of each shapley value
  limevals.append(zip(text, thisvallime))
  limevals.append(thisvallime)

testdata['shapley'] = shapleyvals
print(acc)
#testdata['lime'] = limevals
testdata['predictedval'] = pred
testdata.to_csv(os.path.join(path+'withshapbackup.tsv'), sep='\t', header=0)