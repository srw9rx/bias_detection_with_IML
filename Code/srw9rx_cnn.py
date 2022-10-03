#cnn for multiclass classification, based on NLP class cnn homework and modified
# Necessary packages from yangfeng ji

import sys, os, random, math, sys
import torch, spacy
import numpy as np
from torch import nn
from tqdm import trange

from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from torch.nn import functional as F
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--rpath")
args = vars(parser.parse_args())
path = args['rpath']

## Random seeds, to make the results reproducible from yangfeng ji
seed = 1

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(torch.randn(5))


#from yangfeng ji - define data processing functions
from torchtext.data import Field, ReversibleField, Dataset, TabularDataset, BucketIterator, Iterator

# Download the 'en' module for spaCy using
#  'python -m spacy download en'

spacy_en = spacy.load('en_core_web_sm')

def tokenize_fn(text):
  return [tok.text for tok in tqdm(spacy_en(text))] 


def reader(suffix=".tsv", rpath=args['rpath'], batch_size=8, min_freq=2):
    """
    - suffix: data file suffix
    - rpath: path to the data files
    - batch_size: mini-batch size
    - min_freq: word frequency cutoff, frequency less than min_freq will be removed when building the vocab
    """
    # Utterance Field: text

    TXT = Field(sequential=True, tokenize=tokenize_fn, init_token=None, eos_token=None, lower=True)
    LABEL = Field(sequential=False, unk_token=None, dtype=torch.long, use_vocab=False)
    
    # Create a Dataset instance
    fields = [("label", LABEL), ("text", TXT)]
    trn_data = TabularDataset(os.path.join(rpath+'trn'+suffix), format="TSV", fields=fields, skip_header=True)
    val_data = TabularDataset(os.path.join(rpath+'val'+suffix), format="TSV", fields=fields, skip_header=True)
    tst_data = TabularDataset(os.path.join(rpath+'tst'+suffix), format="TSV", fields=fields, skip_header=True)
    ood_data = TabularDataset(os.path.join("MBICtest"+suffix), format="TSV", fields=fields, skip_header=True)
    # Split
    # Build vocab using training data
    TXT.build_vocab(trn_data, min_freq=min_freq) # or max_size=10000
    # 
    train_iter, val_iter = BucketIterator.splits((trn_data, val_data), # data
                                                             batch_size=batch_size, # 
                                                             sort=True, # sort_key not specified
                                                             sort_key = lambda x : len(x.text),
                                                             shuffle=True, # shuffle between epochs
                                                             repeat=False)
    test_iter, ood_iter = BucketIterator.splits((tst_data, ood_data), # data
                                                             batch_size=batch_size, # 
                                                             sort=True, # sort_key not specified
                                                             sort_key = lambda x : len(x.text),
                                                             shuffle=True, # shuffle between epochs
                                                             repeat=False)
    return train_iter, val_iter, test_iter, ood_iter, TXT

# Commented out IPython magic to ensure Python compatibility.
#process the dataset into groups - from yangfeng ji
train_iter, val_iter, test_iter, ood_iter, txtfield = reader(suffix=".tsv", rpath=path, batch_size=8, min_freq=1)
vocab_size = len(txtfield.vocab)
# print(txtfield.vocab.freqs)
print("Vocab size = {}".format(vocab_size))
pad = txtfield.vocab.stoi[txtfield.pad_token]

print("[TRAIN]:%d (dataset:%d)\t[VAL]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
    % (len(train_iter), len(train_iter.dataset),
    len(val_iter), len(val_iter.dataset),
    len(test_iter), len(test_iter.dataset)))
print("[vocab]:%d" % (vocab_size))

class SimpleCNN(nn.Module):
  #init from yangfeng ji
    def __init__(self, vocab_size, embed_size, drop_rate=0.0,
                 class_size=3, kernel_sizes=[2,3,4],
                 dropout=0.0, using_cuda=True, pad=None):
        super(SimpleCNN, self).__init__()
        # ---------------------------------
        # Configuration
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.class_size = class_size # number of classes
        self.kernel_sizes = kernel_sizes # a list of kernel sizes
        self.using_cuda = using_cuda
        self.pad = pad # index of <pad> in vocab
        # ---------------------------------
        # Model Arch
        # Remove the following line and define some parameters for the CNN model here
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=pad) #shouldn't have to change
        self.fc = nn.Linear(3*embed_size, class_size, bias=True) #we want this layer to be 3*E


    def forward(self, batch):

      #input layer
      label = batch.label
      input = batch.text
      inputembed = self.embed(input).permute(1,2,0) #embed the input and change tensor to proper size

      #convolutional filters 1
      batch1c = torch.nn.Conv1d(in_channels = self.embed_size, out_channels = self.embed_size, kernel_size = 2, padding=1) #convolutional filters
      batch1 = batch1c(inputembed)
      batch1t = torch.tanh(batch1) #nonlinear activation function
      batch1m = torch.max(batch1t, dim=2)[0] #max of batch 1, remove extra dim

      #convolutional filters 2
      batch2c = torch.nn.Conv1d(in_channels = self.embed_size, out_channels = self.embed_size, kernel_size = 2, padding=1)
      batch2= batch2c(inputembed)
      batch2t = torch.tanh(batch2) #nonlinear activation function
      batch2m = torch.max(batch2t, dim =2)[0] #max of batch 2, remove extra dim

      #convolutional filters 3
      batch3c = torch.nn.Conv1d(in_channels = self.embed_size, out_channels = self.embed_size, kernel_size = 2, padding=1)
      batch3 = batch3c(inputembed)
      batch3t = torch.tanh(batch3) #nonlinear activation function
      batch3m = torch.max(batch3t, dim=2)[0] #max of batch 3, remove extra dim

      #concatenation
      maxpooled = torch.cat((batch1m, batch2m, batch3m), dim=1) #3BxC

      logit = self.fc(maxpooled)  #Bx3E
      logprob = F.log_softmax(logit, dim=1)
      loss = F.cross_entropy(logprob, label)
      return loss, logprob
    def predict_proba(self, batch):
        model.eval()
        # records
        val_loss, val_batch = 0, 0
        total_example, correct_pred = 0, 0
        # iterate all the mini batches for evaluation
        for b, batch in enumerate(data_iter):
            # Forward: prediction
            loss, logprob = model(batch)
            # 
            val_batch += 1
            val_loss += loss
            # Argmax
            max_logprob, pred_label = torch.max(logprob, -1)
            correct_pred += (pred_label==batch.label).sum()
            total_example += batch.label.size()[0]
        acc = (1.0*correct_pred)/total_example
        # print("val_batch = {}".format(val_batch))
        return acc


def batch_train(batch, model, optimizer):
    """ Training with one batch
    - batch: a min-batch of the data
    - model: the defined neural network
    - optimizer: optimization method used to update the parameters
    """
    # set in training mode
    model.train()
    # initialize optimizer
    optimizer.zero_grad()
    # forward: prediction
    loss, _ = model(batch)
    # backward: gradient computation
    loss.backward()
    # norm clipping, in case the gradient norm is too large
    clip_grad_norm(model.parameters(), grad_clip)
    # gradient-based update parameter
    optimizer.step()
    return model, loss.item()

def eval(data_iter, model):
    """ Evaluate the model with the data
    data_iter: the data iterator 
    model: the defined model
    """
    # set in the eval model, which will trun off the features only used for training, such as droput
    model.eval()
    # records
    val_loss, val_batch = 0, 0
    total_example, correct_pred = 0, 0
    # iterate all the mini batches for evaluation
    for b, batch in enumerate(data_iter):
        # Forward: prediction
        loss, logprob = model(batch)
        # 
        val_batch += 1
        val_loss += loss
        # Argmax
        max_logprob, pred_label = torch.max(logprob, -1)
        correct_pred += (pred_label==batch.label).sum()
        total_example += batch.label.size()[0]
    acc = (1.0*correct_pred)/total_example
    # print("val_batch = {}".format(val_batch))
    return (val_loss/val_batch), acc

# -----------------------------------
# 1. Random seed
torch.manual_seed(seed)

# ------------------------------------
# 2. Define the model and optimizer
# 'ffn': Feed-forward network; 'cnn': Convolutional neural network
model_name = 'cnn' 
if model_name == 'ffn':
    model =  model = NeuralClassifier(vocab_size, embed_size=64, drop_rate=0.0, class_size=2)
elif model_name == 'cnn':
    model = SimpleCNN(vocab_size, embed_size=64, drop_rate=0.0, class_size=2)
else:
    raise ValueError("Unrecognized model name")
# optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-1, weight_decay=0)
# the norm of grad clipping
grad_clip = 1.0

# ------------------------------------
# 3. Define the numbers of training epochs and validation steps
epoch, val_step = 10, 50

# ------------------------------------
# 4. Training iterations
TrnLoss, ValLoss, ValAcc = [], [], []
total_batch = 0
for e in trange(epoch, desc="epoch "):
    # print(e)
    for b, batch in enumerate(train_iter):
        total_batch += 1
        # Update parameters with one batch
        model, loss = batch_train(batch, model, optimizer)
        # Compute validation loss after each val_step
        if total_batch % val_step == 0:
            val_loss, val_acc = eval(test_iter, model)
            ValLoss.append(float(val_loss))
            ValAcc.append(float(val_acc))
            TrnLoss.append(float(loss))
print("The best validation accuracy = {:.4}".format(max(ValAcc)))

#plt.plot(range(len(TrnLoss)), TrnLoss, color="red", label="Training Loss") # Training loss
#plt.plot(range(len(ValLoss)), ValLoss, color="blue", label="Develoopment Loss") # Val loss
#plt.xlabel("Steps")
#plt.ylabel("NLL")
#plt.legend()
#plt.show(block=True)

torch.save(model.state_dict(), os.path.join(path+'.pth'))
