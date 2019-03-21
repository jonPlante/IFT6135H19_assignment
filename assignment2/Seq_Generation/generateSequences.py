import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy
np = numpy

# NOTE ==============================================
# This is where your models are imported
from models import RNN, GRU 

# Use the GPU if you have one
if torch.cuda.is_available():
	print("Using the GPU")
	device = torch.device("cuda") 
else:
	print("WARNING: You are about to run on cpu, and this will likely run out \
		of memory. \n You can try setting batch_size=1 to reduce memory usage")
	device = torch.device("cpu")
	
	
###############################################################################
#
# 
# DATA LOADING & PROCESSING
#
###############################################################################

# HELPER FUNCTIONS
def _read_words(filename):
    with open(filename, "r") as f:
      return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

# Processes the raw data from text files
def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word


data='data'
# LOAD DATA
print('Loading data from '+data)
raw_data = ptb_raw_data(data_path=data)
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
vocab_size = len(word_to_id)
print('  vocabulary size: {}'.format(vocab_size))


embSize=[200,200]
batchSize=[20,20]
dropOut=[0.35,0.35]
hiddenSize=[1500,1500]
model_types=['RNN','GRU']
numLayers=[2,2]
seqLen=[35,35]
seq_len=[35,70]
samples=10
path=['best_params_RNN.pt','best_params_GRU.pt']




for m in range(len(model_types)):
	for s in range(len(seq_len)):
		print('Processing model: '+model_types[m]+' seq_len: '+str(seq_len[s])+'\n')
		if model_types[m]=='RNN':
			model = RNN(emb_size=embSize[m], hidden_size=hiddenSize[m], 
					seq_len=seqLen[m], batch_size=batchSize[m],
					vocab_size=vocab_size, num_layers=numLayers[m], 
					dp_keep_prob=dropOut[m])
		else:
			model =GRU(emb_size=embSize[m], hidden_size=hiddenSize[m], 
					seq_len=seqLen[m], batch_size=batchSize[m],
					vocab_size=vocab_size, num_layers=numLayers[m], 
					dp_keep_prob=dropOut[m])
		model.load_state_dict(torch.load(path[m]))
		model = model.to(device)
		hidden = nn.Parameter(torch.zeros(numLayers[m],samples,hiddenSize[m])).to(device)
		input=torch.ones(10000)*1/1000
		input=torch.multinomial(input,samples).to(device)
		output=model.generate(input, hidden, seq_len[s])
		print('Saving generated samples')
		fid=open(model_types[m]+'_' +str(seq_len[s])+'.txt','w')
		for i in range(samples):
			for j in range(seq_len[s]):
				fid.write(id_2_word.get(output[j,i].item())+' ')
			fid.write('\n')
		fid.close()