# -*- coding: utf-8 -*-
# ====================================================================
# ---------------------------< WELCOME >------------------------------
# ====================================================================
""" LSTM Training"""

# To Run on opuntia:
# 1. srun -A kakadiaris -t 1:00:00 -n 20 -p gpu --gres=gpu:1 -N 1 --pty /bin/bash -l
# 2. module load CUDA/9.2.148
# 3. source Envv2/bin/activate
# 4. cd LSTM_Sentinet

import json
import time
import random

import numpy as np
import warnings
import lstmhelper as helper

import mxnet.ndarray as nd
import mxnet as mx
from mxnet import gluon, autograd
import gluonnlp as nlp
import multiprocessing as mp

# from mxnet.symbol import broadcast_mul

from mxnet.ndarray import broadcast_mul, broadcast_hypot, dot
random.seed(123)
np.random.seed(123)
mx.random.seed(123)
warnings.filterwarnings('ignore')

grad_clip = None                       # Set to True to apply gradiant clipping during training
log_interval = 30                      # Print after this many batches
context = mx.gpu()
# context = mx.cpu()




# a = mx.nd.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
# print(a.shape)
# x = mx.sym.Variable('x', shape=(10, ), init=mx.init.Constant(0.25))
#
# print(x.asnumpy())

# print(x.list_outputs())
# print(x.get_internals())
# print(x.get_children())
# print(x.list_arguments())
# print(x.get_params())
# print(a / nd.sqrt(nd.dot(a, a))[0])
# result = nd.sqrt(nd.sum(broadcast_mul(a, a)))
# print(result)

# exit(0)
# --------------------------------------------------------------------
# -------------------< Definitions & Functions >----------------------
# --------------------------------------------------------------------


class MeanPoolingLayer(gluon.HybridBlock):
    """A block for mean pooling of encoder features"""
    def __init__(self, prefix=None, params=None):
        super(MeanPoolingLayer, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, data, valid_length): # pylint: disable=arguments-differ
        """Forward logic"""
        # Data will have shape (T, N, C)
        masked_encoded = F.SequenceMask(data,
                                        sequence_length=valid_length,
                                        use_sequence_length=True)
        agg_state = F.broadcast_div(F.sum(masked_encoded, axis=0),
                                    F.expand_dims(valid_length, axis=1))
        return agg_state


# class NormalizationHybridLayer(gluon.HybridBlock):
#
#     def __init__(self):
#         super(NormalizationHybridLayer, self).__init__()
#
#     def hybrid_forward(self, F, x):
#         return x / F.sqrt(F.dot(x, x))[0]


class SentimentNet(gluon.HybridBlock):
    """Network for sentiment analysis."""
    def __init__(self, dropout, prefix=None, params=None):
        super(SentimentNet, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.embedding = None           # will set with lm embedding later
            self.encoder = None             # will set with lm encoder later
            self.agg_layer = MeanPoolingLayer()
            self.output = gluon.nn.HybridSequential()
            with self.output.name_scope():
                self.output.add(gluon.nn.Dense(50, in_units=200, activation='sigmoid'))
                self.output.add(gluon.nn.Dropout(dropout))
                self.output.add(gluon.nn.Dense(10, activation='relu'))
                # self.output.add(gluon.nn.Dense(10, activation='relu'))
                # self.output.add(NormalizationHybridLayer())
                self.output.add(gluon.nn.Dense(1, flatten=False))

    def hybrid_forward(self, F, data, valid_length):    # pylint: disable=arguments-differ
        encoded = self.encoder(self.embedding(data))    # Shape(T, N, C)
        agg_state = self.agg_layer(encoded, valid_length)
        out = self.output(agg_state)
        return out


# ------------------------------------
# -------- Preprocessing Trio --------
def preprocess_dataset(dataset):
    """Docstring"""
    start = time.time()
    with mp.Pool() as pool:
        # Each sample is processed in an asynchronous manner.
        dataset = gluon.data.SimpleDataset(pool.map(preprocess, dataset))
        lengths = gluon.data.SimpleDataset(pool.map(get_length, dataset))
    end = time.time()
    print('   Done Preparing Data: Time={:.2f}s, # of Sentences={}'.format(end - start, len(dataset)))
    return dataset, lengths


def preprocess(x):
    """Docstring"""
    data, label = x
    label = int(label)
    data = vocab[length_clip(data.split('\t'))]
    return data, label


def get_length(x):
    """Docstring"""
    return float(len(x[0]))
# ------------- End Trio --------------
# -------------------------------------


# Training Data Loaders
def get_dataloader():
    """Docstring"""
    # Construct the DataLoader
    # Pad data, stack label and lengths
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, ret_length=True),
        nlp.data.batchify.Stack(dtype='float32'))
    batch_sampler = nlp.data.sampler.FixedBucketSampler(
        train_data_lengths,
        batch_size=batch_size,
        num_buckets=bucket_num,
        ratio=bucket_ratio,
        shuffle=True)
    train_dataloader = gluon.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=batch_sampler,
        batchify_fn=batchify_fn)
    test_dataloader = gluon.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        batchify_fn=batchify_fn)
    return train_dataloader, test_dataloader


def evaluate(net, dataloader, context):
    """Testing function"""
    loss = gluon.loss.SigmoidBCELoss()
    total_L = 0.0
    total_sample_num = 0
    total_correct_num = 0
    start_log_interval_time = time.time()
    print('  Begin Testing...')

    for i, ((data, valid_length), label) in enumerate(dataloader):

        data = mx.nd.transpose(data.as_in_context(context))
        valid_length = valid_length.as_in_context(context).astype(np.float32)
        label = label.as_in_context(context)
        output = net(data, valid_length)
        L = loss(output, label)
        pred = (output > 0.5).reshape(-1)
        total_L += L.sum().asscalar()
        total_sample_num += label.shape[0]
        total_correct_num += (pred == label).sum().asscalar()
        if (i + 1) % log_interval == 0:
            print('    >> [Batch {}/{}] elapsed {:.2f} s'.format(
                i + 1, len(dataloader),
                time.time() - start_log_interval_time))
            start_log_interval_time = time.time()
    avg_L = total_L / float(total_sample_num)
    acc = total_correct_num / float(total_sample_num)
    return avg_L, acc


def train(net, context, epochs):

    """Primary Training Function"""
    trainer = gluon.Trainer(net.collect_params(), 'ftml', {'learning_rate': learning_rate})
    loss = gluon.loss.SigmoidBCELoss()

    # This is only necessary if gradient clipping is active
    parameters = net.collect_params().values()

    # Training/Testing
    for epoch in range(epochs):

        # Epoch training stats
        start_epoch_time = time.time()
        epoch_sent_num, epoch_wc, epoch_L = 0, 0, 0.0
        # Log interval training stats
        start_log_interval_time = time.time()
        log_interval_wc, log_interval_sent_num, log_interval_L = 0, 0, 0.0
        for i, ((data, length), label) in enumerate(train_dataloader):
            L = 0
            wc = length.sum().asscalar()
            log_interval_wc += wc
            epoch_wc += wc
            log_interval_sent_num += data.shape[1]
            epoch_sent_num += data.shape[1]
            # Feed forward & Back-Propagate
            with autograd.record():
                output = net(data.as_in_context(context).T,
                             length.as_in_context(context)
                                   .astype(np.float32))
                L = L + loss(output, label.as_in_context(context)).mean()
            L.backward()

            # Clip gradient if active
            if grad_clip:
                gluon.utils.clip_global_norm(
                    [p.grad(context) for p in parameters],
                    grad_clip)
            # Update parameters
            trainer.step(1)
            log_interval_L += L.asscalar()
            epoch_L += L.asscalar()
            # Print Log every log_value
            if (i + 1) % log_interval == 0:
                # print("%05s. %s %10s" % (str(i), str(label), str(data.shape)))
                print(
                    '    >> [Epoch {} Batch {}/{}] elapsed {:.2f} s, '
                    'avg loss {:.6f}, throughput {:.2f}K wps'.format(
                        epoch, i + 1, len(train_dataloader),
                        time.time() - start_log_interval_time,
                        log_interval_L / log_interval_sent_num, log_interval_wc
                        / 1000 / (time.time() - start_log_interval_time)))
                # Clear log interval training stats
                start_log_interval_time = time.time()
                log_interval_wc = 0
                log_interval_sent_num = 0
                log_interval_L = 0
        end_epoch_time = time.time()
        # Run Testing after each epoch is are complete
        test_avg_L, test_acc = evaluate(net, test_dataloader, context)
        print('        *** [Epoch {}] train avg loss {:.6f}, test acc {:.5f}, '
              'test avg loss {:.6f},\nthroughput {:.2f}K wps\n'.format(
                  epoch, epoch_L / epoch_sent_num, test_acc, test_avg_L,
                  epoch_wc / 1000 / (end_epoch_time - start_epoch_time)))


# --------------------------------------------------------------------
# ---------------------------< Script >-------------------------------
# --------------------------------------------------------------------

helper.welcomebanner(' LSTM TRAINING ')

argparser = helper.lstm_argparser()          # Initialize argument parser
opts = argparser.parse_args()                # Parse input arguments

# Arguments
epochs = opts.epochs[0]
dropout = opts.dropout[0]                    # Dropout rate  0 - 1.0
batch_size = opts.batchsize[0]               # Batch Size
learning_rate = opts.learningrate[0]         # Default 0.005
vocab_path = opts.vocabpath[0]               # Vocabulary file
bucket_num, bucket_ratio = 10, 0.2           # Bucketing Parameters (training only)

# (Read this if you want to know how bucketing works:
# https://mxnet.incubator.apache.org/versions/master/faq/bucketing.html)

trainingpath = opts.trainpath[0]
testingpath = opts.testpath[0]
#     'data/trump/run1/trumpTest.json'
# 'data/trump/run1/trumpTrain.json' # Training data path
# testingpath = 'data/trump/run1/trumpTest.json'   # Testing data path

# 1. Load Pre-Trained Model
#       Every time this trains, it starts from these pretrained features.
#       We may, if we please, load the features from one of our own trained models
# my_vocab =  helper.loadvocabdictionary(vocab_path)
my_vocab = helper.loadvocabdictionary('data/dicts/' + opts.vocabpath[0])
vocab = nlp.Vocab(my_vocab)

# ** Note: _ is for the vocab, which is discarded (we use the the above vocab)
lm_model, _ = nlp.model.get_model(name='standard_lstm_lm_200', dataset_name='wikitext-2',
                                  pretrained=True, ctx=context, dropout=dropout)

myLSTMnet = SentimentNet(dropout=dropout)                     # Initialize the model
myLSTMnet.embedding = lm_model.embedding                      # Transfer pretrained embedding features
myLSTMnet.encoder = lm_model.encoder                          # Transfer pretrained encoder features
myLSTMnet.hybridize()
myLSTMnet.output.initialize(mx.init.Xavier(), ctx=context)    # Init output layers

# 2. Initialize Tokenizer
# tokenizer = nlp.data.SpacyTokenizer('en')                   # Uncomment when using spacy
length_clip = nlp.data.ClipSequence(500)                      # Don't want to clip longer than 500 (too long for LSTM to model)

# 3. Open Testing & Training Data
with open(trainingpath) as json_file:
    train_dataset = json.load(json_file)
    json_file.close()
with open(testingpath) as json_file:
    test_dataset = json.load(json_file)
    json_file.close()


# 4. ------< Pre-process Data >--------
print('   ... Pre-processing...')
train_dataset, train_data_lengths = preprocess_dataset(train_dataset)
test_dataset, test_data_lengths = preprocess_dataset(test_dataset)
train_dataloader, test_dataloader = get_dataloader()

# 5. ----------<  Train >--------------
train(myLSTMnet, context, epochs)


# 6.----------< Save Model >-----------

print("    >> Exporting Model: %s" % opts.modelname[0])
myLSTMnet.export(opts.modelname[0])


# --------------------------------------------------------------------
# -----------------------------< End >--------------------------------
# --------------------------------------------------------------------

# 3. Load IMDB Dataset
# train_dataset, test_dataset = ...
# [nlp.data.IMDB(root='data/imdb', segment=segment) ...
# for segment in ('train', 'test')]

# Feedforward example:
# sentence = ['make', 'america', 'great', 'again']
# output = net(mx.nd.reshape(
#         mx.nd.array(vocab[sentence], ctx=context),
#         shape=(-1, 1)), mx.nd.array([4], ctx=context)).sigmoid()
#

