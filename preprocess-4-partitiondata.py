# -*- coding: utf-8 -*-
# ====================================================================
# ---------------------------< WELCOME >------------------------------
# ====================================================================
"""This is the last step, partition the data &
creates partitioned json files to use in the LSTM Network"""

import random
import numpy as np
from numpy import ndarray as nd
import math
import lstmhelper as helper
import matplotlib.pyplot as plt

random.seed(13)     # For random sample


# --------------------------------------------------------------------
# --------------------------< Functions >-----------------------------
# --------------------------------------------------------------------

def partition(input_data, part_scheme):
    """Partition the data"""
    # Separate into  class 0 and 1 for even partitioning
    neg_class = [item for item in input_data if int(item[1]) == 0]
    pos_class = [item for item in input_data if int(item[1]) == 1]
    # Define partitions
    t1 = int(part_scheme[0] * len(neg_class))
    t2 = t1 + int(part_scheme[1] * len(neg_class))
    # Partition
    train_part = neg_class[0:t1] + pos_class[0:t1]
    test_part = neg_class[t1:t2] + pos_class[t1:t2]
    val_part = neg_class[t2:] + pos_class[t2:]

    return train_part, test_part, val_part


def sample_and_partition(shorter_data, longer_data, part_scheme):
    """Whichever dataset is shorter, give it as the first argument
    In this case, the Trump dataset is far shorter than the Youtube"""



    sample = random.sample(list(enumerate(longer_data)), len(shorter_data))


    sample_sentences = [item[1] for item in sample]
    sample_indexes = [item[0] for item in sample]
    combined_sample = shorter_data + sample_sentences
    # print("Combined Sample Length: %d " % len(combined_sample))

    ttrain, ttest, tval = partition(combined_sample, part_scheme)

    leftover = longer_data
    for index in sorted(sample_indexes, reverse=True):
        # print(leftover[index])
        del (leftover[index])



    return ttrain, ttest, tval, leftover


def partition_summary(data_partition, name):
    """Provides a summary of the partition including lengths & most
    frequent words
    """
    print("\n    >> %s" % name)
    plen = len(data_partition)
    lengths = [len(item[0].split('\t')) for item in data_partition]
    mean_len = sum([len(item[0].split('\t')) for item in data_partition]) / plen

    neg_class = [item for item in data_partition if int(item[1]) == 0]
    pos_class = [item for item in data_partition if int(item[1]) == 1]
    neg_lengths = [len(item[0].split('\t')) for item in neg_class]
    pos_lengths = [len(item[0].split('\t')) for item in pos_class]
    print("    >> Number of Comments: %d" % plen)
    print("    >> Avg Comment Length: %f" % mean_len)
    print("    >> Negative Samples: %d  -- Positive Samples: %d" % (len(neg_class), len(pos_class)))
    print("    >> Min: %d -- Max: %d" % (min(lengths), max(lengths)))

    return pos_lengths, neg_lengths


def partition_by_length(data0, data1, w, limit):
    """docstring"""

    samples0 = []
    samples1 = []

    for a in range(1, limit, w):

        b = a + 10
        # Take a random sample at each level
        comment_at_length0 = [comment for comment in data0 if len(comment[1][0]) in range(a, b)]
        comment_at_length1 = [comment for comment in data1 if len(comment[1][0]) in range(a, b)]
        sample = random.sample(comment_at_length0, len(comment_at_length1))
        samples0.append(sample)
        samples1.append(comment_at_length1)
    return samples0, samples1


def do_even_partitions(input_samples, part_scheme):
    """docstring"""
    train_part, test_part, val_part = [], [], []
    indexes = []
    partition_list = []
    trainlist, testlist, vallist, indexlist = [], [], [], []
    for b, bucket_level in enumerate(input_samples):

        # print(class_partition(c0, part_scheme))
        train, test, val, index = class_partition(bucket_level, part_scheme)
        trainlist.extend(train)
        testlist.extend(test)
        vallist.extend(val)
        indexlist.extend(index)

    return trainlist, testlist, vallist, indexlist


def class_partition(indata, part_scheme):
    """docstring"""

    t1 = int(part_scheme[0] * len(indata))
    t2 = t1 + int(part_scheme[1] * len(indata))
    train = [item[1] for item in indata[0:t1]]
    test = [item[1] for item in indata[t1:t2]]
    val = [item[1] for item in indata[t2:]]
    index = [item[0] for item in indata]
    return train, test, val, index


def flatten_list(list2d):
    """docstring"""
    temp = []
    for i in list2d:
        temp.extend(i)
    return temp


# --------------------------------------------------------------------
# ----------------------------< Script >------------------------------
# --------------------------------------------------------------------

# Training testing and validation partitioning
train, test, val = 0.6, 0.3, 0.1
partition_scheme = (train, test, val)
run = 'run2/'

# data_set = 'trump'
# out1 = 'data/jsondata/trump/'
# data_path = 'data/jsondata/splitTrump-400k.json'
# train_path, test_path, val_path = 'trumpTrain.json', 'trumpTest.json', 'trumpVal.json'
# #
data_set = 'bre'
out1 = 'data/jsondata/bre/'
data_path = 'data/jsondata/bre-comments-400k.json'
train_path, test_path, val_path = 'breTrain.json', 'breTest.json', 'breVal.json'

# ~~~~~~~~~~~~~< (1) LOAD DATA >~~~~~~~~~~~~~~
ground_truth_data = helper.fopenjson(data_path)
youtube_comment_data = helper.fopenjson('data/jsondata/youtube_good_data-400k.json')


gt = [item for item in enumerate([[d[0].split('\t'), d[1]] for d in ground_truth_data])]
yt = [item for item in enumerate([[d[0].split('\t'), d[1]] for d in youtube_comment_data])]
#


# Outputs have on level for each bucket level
samples0, samples1 = partition_by_length(yt, gt, 10, 250)

# At each bucket level select the proper proportion
train0, test0, val0, indices = do_even_partitions(samples0, partition_scheme)
train1, test1, val1, _ = do_even_partitions(samples1, partition_scheme)


train0_lengths, train1_lengths = [len(i[0]) for i in train0], [len(i[0]) for i in train1]
test0_lengths, test1_lengths = [len(i[0]) for i in test0], [len(i[0]) for i in test1]
val0_lengths, val1_lengths = [len(i[0]) for i in val0], [len(i[0]) for i in val1]


leftover_comments = youtube_comment_data
for index in sorted(indices, reverse=True):
    del(leftover_comments[index])


# # 2. Partition the data
# train, test, val, leftover_comments = sample_and_partition(ground_truth_data, youtube_comment_data, partition_scheme)
# #
# # 3. Output verification for various partitions
# training = partition_summary(train, "Training")
# testing = partition_summary(test, 'Testing')
# validation = partition_summary(val, 'Validation')
# leftover = partition_summary(leftover_comments, 'Leftover Comments')
# ground_truth = partition_summary(ground_truth_data, 'Ground Truth Data')
# all_data_combined = partition_summary(ground_truth_data + youtube_comment_data, 'All Data')


# partition_by_length(data0, data1, w, limit):


fig1, f1_axes = plt.subplots(ncols=3, nrows=1, num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')


clr = ['tab:blue', 'tab:red']
BINS = np.arange(1, 201, 10).tolist()
f1_axes[0].hist((train0_lengths, train1_lengths), bins=BINS, color=clr, stacked=False)
f1_axes[1].hist((test0_lengths, test1_lengths), bins=BINS, color=clr, stacked=False)
f1_axes[2].hist((val0_lengths, val1_lengths), bins=BINS, color=clr, stacked=False)
# f1_axes[1].hist(testing, bins=BINS, color=clr, stacked=True)
# f1_axes[1].hist(testing, bins=BINS, color=clr, stacked=True)
# f1_axes[2].hist(validation, bins=BINS, color=clr, stacked=True)


f1_axes[1].set_xlabel('Sentence Length')
f1_axes[0].set_ylabel('Frequency')

f1_axes[0].set_title('Training')
f1_axes[0].set_ylim(1, 1000)
f1_axes[1].set_title('Testing')
f1_axes[1].set_ylim(1, 500)
f1_axes[2].set_title('Validation')
f1_axes[2].set_ylim(1, 250)

# f1_axes[1][0].hist(ground_truth, bins=BINS, color=['r', 'b'], stacked=True)
# f1_axes[1][1].hist(leftover, bins=BINS, color=['r', 'b'], stacked=True)
# f1_axes[1][2].hist(all_data_combined, bins=BINS, color=['r', 'b'], stacked=True)
# f1_axes[1][0].set_title(data_set)
# f1_axes[1][1].set_title('Youtube')
# f1_axes[1][2].set_title(data_set + ' & Youtube')

fig1.suptitle("Partitioned Data Overview", fontsize=16, y=0.99)
f1_axes[1].legend(('Youtube', 'Personal'))
plt.show()

train = train0 + train1
test = test0 + test1
val = val0 + val1

train = [['\t'.join(item[0]), item[1]] for item in train0 + train1]
test = [['\t'.join(item[0]), item[1]] for item in test0 + test1]
val = [['\t'.join(item[0]), item[1]] for item in val0 + val1]

helper.fwritejson(train, out1+run+train_path)
helper.fwritejson(test, out1+run+test_path)
helper.fwritejson(val, out1+run+val_path)
helper.fwritejson(leftover_comments, out1+run+'/youtube_leftover_comments.json')

exit(0)

# ====================================================================
# -----------------------------< End >--------------------------------
# ====================================================================

