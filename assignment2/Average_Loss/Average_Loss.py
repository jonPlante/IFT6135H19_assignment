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
import matplotlib.pyplot as plt

# NOTE ==============================================
# This is where your models are imported
from models import RNN, GRU 
from models import make_model as TRANSFORMER



# Set the random seed manually for reproducibility.
torch.manual_seed(1111)

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

# Yields minibatches of data
def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)


class Batch:
    "Data processing for the transformer. This class adds a mask to the data."
    def __init__(self, x, pad=-1):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."

        def subsequent_mask(size):
            """ helper function for creating the masks. """
            attn_shape = (1, size, size)
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
            return torch.from_numpy(subsequent_mask) == 0

        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask

data='data'
# LOAD DATA
print('Loading data from '+ data)
raw_data = ptb_raw_data(data_path=data)
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
vocab_size = len(word_to_id)
print('  vocabulary size: {}'.format(vocab_size))


###############################################################################
# 
# MODEL SETUP
#
###############################################################################

#

# LOSS FUNCTION
loss_fn = nn.CrossEntropyLoss()


###############################################################################
# 
# DEFINE COMPUTATIONS FOR PROCESSING ONE EPOCH
#
###############################################################################

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    
    This prevents Pytorch from trying to backpropagate into previous input 
    sequences when we use the final hidden states from one mini-batch as the 
    initial hidden states for the next mini-batch.
    
    Using the final hidden states in this way makes sense when the elements of 
    the mini-batches are actually successive subsequences in a set of longer sequences.
    This is the case with the way we've processed the Penn Treebank dataset.
    """
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)


def run_epoch(model, data, type, is_train=False, lr=1.0):
    """
    One epoch of training/validation (depending on flag is_train).
    """
    model.eval()
    epoch_size = ((len(data) // model.batch_size) - 1) // model.seq_len
    
    average_loss=np.zeros((1,model.seq_len))
    iters=0
    # LOOP THROUGH MINIBATCHES
    for step, (x, y) in enumerate(ptb_iterator(data, model.batch_size, model.seq_len)):
        if type != 'TRANSFORMER':
            hidden = model.init_hidden()
            hidden = hidden.to(device)
            inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
            model.zero_grad()
            hidden = repackage_hidden(hidden)
            outputs, hidden = model(inputs, hidden)
        else:
            batch = Batch(torch.from_numpy(x).long().to(device))
            model.zero_grad()
            outputs = model.forward(batch.data, batch.mask).transpose(1,0)
        targets = torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
        for t in range(outputs.size(0)):
            tt=torch.squeeze(targets[t,:].view(-1,model.batch_size))
            loss = loss_fn(outputs[t,:,:].contiguous().view(-1, model.vocab_size), tt)
            average_loss[0,t]=average_loss[0,t]+loss.data.item()
        iters +=1
    average_loss=average_loss/iters
    return average_loss


###############################################################################
#
# RUN MAIN LOOP (TRAIN AND VAL)
#
###############################################################################
embSize=[200,200]
batchSize=[20,20,128]
dropOut=[0.35,0.35,0.9]
hiddenSize=[1500,1500,512]
model_types=['RNN','GRU','TRANSFORMER']
numLayers=[2,2,6]
seqLen=[35,35,35]
samples=10
path=['best_params_RNN.pt','best_params_GRU.pt','best_params_TRANS.pt']



total_loss=np.zeros((3,35))
for m in range(len(model_types)):
    print("\n########## Running Main Loop ##########################")
    train_ppls = []
    train_losses = []
    val_ppls = []
    val_losses = []
    best_val_so_far = np.inf
    times = []
    if model_types[m]=='RNN':
        model = RNN(emb_size=embSize[m], hidden_size=hiddenSize[m], 
                    seq_len=seqLen[m], batch_size=batchSize[m],
                    vocab_size=vocab_size, num_layers=numLayers[m], 
                    dp_keep_prob=dropOut[m])
    elif model_types[m]=='GRU':
        model =GRU(emb_size=embSize[m], hidden_size=hiddenSize[m], 
                   seq_len=seqLen[m], batch_size=batchSize[m],
                   vocab_size=vocab_size, num_layers=numLayers[m], 
                   dp_keep_prob=dropOut[m])
    else:
        model=TRANSFORMER(vocab_size=vocab_size,n_units=hiddenSize[m],
                          n_blocks=numLayers[m],dropout=1-dropOut[m])
    model.load_state_dict(torch.load(path[m]))
    model.batch_size=batchSize[m]
    model.seq_len=seqLen[m]
    model.vocab_size=vocab_size
    model = model.to(device)
    
    # MAIN LOOP
    val_loss = run_epoch(model, valid_data,model_types[m])
    total_loss[m,:]=val_loss
    time=np.arange(1,seqLen[m]+1)
    print('Plotting graph...')
    plt.figure()
    plt.plot(time, val_loss.flatten(), label='Val. Loss')
    plt.ylabel('Average loss')
    plt.xlabel('time-step (t)')
    plt.grid(True)
    plt.title('Average loss at each time-step on validation for '+model_types[m])
    plt.savefig(os.path.join(model_types[m]+'_avg_loss.png'))
    plt.clf()

plt.figure()
plt.plot(time, total_loss[0,:].flatten(), label='RNN')
plt.plot(time, total_loss[1,:].flatten(), label='GRU')
plt.plot(time, total_loss[2,:].flatten(), label='TRANSFORMER')
plt.ylabel('Average loss')
plt.xlabel('time-step (t)')
plt.grid(True)
plt.legend()
plt.title('Average loss at each time-step on validation for all models')
plt.savefig(os.path.join('RNN_GRU_TRANS_avg_loss.png'))
plt.clf()
    
    