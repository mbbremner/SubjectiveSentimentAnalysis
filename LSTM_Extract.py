# -*- coding: utf-8 -*-
# ====================================================================
# ---------------------------< WELCOME >------------------------------
# ====================================================================
"""LSTM Feature Extraction
    Features are written to file in the following format:
    index \t label \t comma,delimited,feature,vector
"""

import numpy as np
import random
import time
import re
import io
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
import warnings
import multiprocessing as mp
import lstmhelper as helper

# Housekeeping
context = mx.gpu()

random.seed(123)
np.random.seed(123)
mx.random.seed(123)

warnings.filterwarnings('ignore')
length_clip = nlp.data.ClipSequence(500)

head = 'data/features/'
# Note, these aren't used anymore a command line arg is taken instead
outfiles = ['youtubesigmoid4features10.txt', 'trumpsig4features10.txt', 'trumptrainingsig4features10.txt', 'trumpTESTsig4features10.txt']

# ====================================================================
# --------------------------< Functions >-----------------------------
# ====================================================================


def perform_feature_extraction(jsonpath, net):
    """Feature Extraction Wrapper:
            Opens & Processes data, makes a dataloader to batchify
            and extracts the features by running 1 epoch"""
    print("    >> Extracting Features From: %s" % opts.jsonpath[0])
    temp_data = helper.fopenjson(jsonpath)
    temp_data, temp_lens = preprocess_dataset(temp_data)
    temp_data_loader = get_dataloader(temp_data)
    features, labels, predictions = extract_features(net, temp_data_loader)
    return features, predictions, labels, temp_lens


def extract_features(nnmodel, dataloader):
    """Feature Extraction Function: Called by perform_feature_extraction
        Runs through a single epoch and extracts features for each example
        Returns the labels along with features"""

    print('    >> Extracting features by feed forward ... ')
    start = time.monotonic()

    temp_feature_list, temp_labels, temp_predictions = [], [], []
    for epoch in range(1):
        for i, ((data, length), label) in enumerate(dataloader):
            output = nnmodel(data.as_in_context(context).T,
                         length.as_in_context(context).astype(np.float32))
            temp_feature_list.extend(output[0])
            temp_labels.extend(label)
            temp_predictions.extend(output[1].sigmoid())
    print("    >> ... Done - Time: %5.3f s" % (time.monotonic() - start))
    return temp_feature_list, temp_labels, temp_predictions


# ------------------------------------
# -------- Preprocessing Trio --------
def preprocess_dataset(dataset):
    """Docstring"""
    print("    >> Pre-Processing ...")
    start = time.time()
    with mp.Pool() as pool:
        # Each sample is processed in an asynchronous manner.
        dataset = gluon.data.SimpleDataset(pool.map(preprocess, dataset))
        lengths = gluon.data.SimpleDataset(pool.map(get_length, dataset))
    end = time.time()
    print('        - Done! Preprocessing Time={:.2f}s, #Sentences={}'.format(end - start, len(dataset)))
    return dataset, lengths


def preprocess(x):
    """For the IMDB dataset the label was determined by assigning > 5 to
       positive and otherwise to negative.  I commented that part out and
       left it.
       Where it says thedata.split('\t')), this code used to call the spacy
       tokenizer function, however, the data we use is already tokenized so
       it just needs to be split by the delim"""
    thedata, thelabel = x
    # label = int(label > 5)
    thelabel = int(thelabel)
    thedata = vocab[length_clip(thedata.split('\t'))]
    return thedata, thelabel


def get_length(x):
    """Docstring"""
    return float(len(x[0]))
# -------- Preprocessing Trio ---------
# -------------------------------------


def get_dataloader(inputdata):
    """A simplified data loader for feature extraction"""
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, ret_length=True),
        nlp.data.batchify.Stack(dtype='float32'))
    thedataloader = gluon.data.DataLoader(
        batch_size=batch_size,
        dataset=inputdata,
        batchify_fn=batchify_fn)
    return thedataloader


def fwrite_features(savepath, inputdata):
    """ Standard format for writing features to file:
            ID\length\ttLABEL\tprediction\tCOMMA,DELIMITED,FEATURE,VECTOR"""

    features, predictions, labels, lengths = inputdata

    f = io.open(savepath, 'w', encoding='utf-8')
    for v, vector in enumerate(features):
        prediction = str(predictions[v].asnumpy().tolist()[0])
        label = str(int(labels[v].asnumpy()[0]))
        vector = ','.join([str(item) for item in vector.asnumpy().tolist()])
        f.write(str(v+1) + '\t' + str(int(lengths[v])) + '\t' + prediction + '\t' + label + '\t' + vector + '\n')
    f.close()
    print("    >> Saved to: %s \n" % savepath)


# ====================================================================
# ---------------------------< Script >-------------------------------
# ====================================================================

helper.welcomebanner(' LSTM FEATURE EXTRACTION ')

argparser = helper.lstm_argparser()            # Initialize argument parser
opts = argparser.parse_args()                  # Parse input arguments

# Arguments
model, batch_size = opts.modelname[0], opts.batchsize[0]
feature_path, json_path = opts.featurepath[0], opts.jsonpath[0]
act = opts.acttype[0]

# 1. ------------- Load The LSTM Network --------------

param_file = model + '-0000.params'
LSTMsym = mx.sym.load(model + '-symbol.json')
LSTMinternals = LSTMsym.get_internals()

# Output Layers
feature_layer = 'hybridsequential0_dense1_'+act+'_fwd'
featureLayer = [item for item in LSTMinternals if re.search(feature_layer, str(item))]
# featureLayer = [item for item in LSTMinternals if re.search('hybridsequential0_dense1_sigmoid_fwd', str(item))]
prediction_layer = [item for item in LSTMinternals if re.search('hybridsequential0_dense2_fwd>', str(item))]
output_layers = mx.symbol.Group([featureLayer[0], prediction_layer[0]])

# Input Layer
input_layers = [mx.sym.var('data0'), mx.sym.var('data1')]

# Model Model Model!
myLSTMnet = gluon.nn.SymbolBlock(outputs=output_layers, inputs=input_layers)    # Initialize net w/ desired output
myLSTMnet.collect_params().load(param_file, ctx=context, ignore_extra=True)     # Load model parameters


# 2. -------------- Feature Extraction ----------------

# a. Load vocab and insert into vocabulary object
vpath = opts.vocabpath[0]
my_vocab = helper.loadvocabdictionary(vpath)
vocab = nlp.Vocab(my_vocab)

# b. Extract Features
extracted = perform_feature_extraction(json_path, myLSTMnet)
# features, labels, lengths
print("    >> Writing to File: ")

fwrite_features('data/features/' + feature_path, extracted)

exit(0)


# ====================================================================
# -----------------------------< End >--------------------------------
# ====================================================================


