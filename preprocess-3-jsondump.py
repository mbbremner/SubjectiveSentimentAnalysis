# -*- coding: utf-8 -*-
# ====================================================================
# ---------------------------< WELCOME >------------------------------
# ====================================================================

""" (1) Open the split sentences, (2) append a ground truth tag
    (3) dump to json file
    This script likely does not need to exist and could be made into a
    function within the previous script
    Modify the file names as necessary.  The input should be the output
    processed comments of step (2)
    """

import lstmhelper as helper
import matplotlib.pyplot as plt
import numpy as np

class1_out, class0_out = [], []

# --------------------------------------------------------------------
# --------------------------< Functions >-----------------------------
# --------------------------------------------------------------------


def data_length_histogram(comment_lines):

    lengths = [len(i) for i in comment_lines]
    print(lengths)

    fig1, ax = plt.subplots(ncols=1, nrows=1, num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
    # BINS = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,150,200,500,1000,2000]
    BINS = np.arange(0,100,5).tolist() + np.arange(100, 1000, 50).tolist()
    ax.hist(lengths, bins=BINS, color='b')
    plt.show()
    exit(0)


# --------------------------------------------------------------------
# ----------------------------< Script >------------------------------
# --------------------------------------------------------------------

va, vb = 3, 250
# va, vb = 10, 80

in0 = 'data/comments-good-youtube-400k.txt'
# in1 = 'data/trump-tweets-processed.txt'
in1 = 'data/bre-comments-processed-400k.txt'

out0 = 'data/jsondata/youtube_good_data-400k.json'
# out1 = 'data/jsondata/splitTrump-400k.json'
out1 = 'data/jsondata/bre-comments-400k.json'


# Read Data & Split
class0_split = [line.strip().split('\t') for line in helper.freadlist(in0)]
class1_split = [line.strip().split('\t')[0].split(',') for line in helper.freadlist(in1)]




# Constrain Sentences to a particular length
class1_restrict = [a for a in class1_split if len(a) in range(va, vb)]
class0_restrict = [a for a in class0_split if len(a) in range(va, vb)]

print(len(class1_split), len(class0_split))
print(len(class1_restrict), len(class0_restrict))

# Smsch together & tack on a ground truth
for line in class0_restrict:
    line = '\t'.join([item.lower() for item in line])
    class0_out.append([line, 0])

for line in class1_restrict:
    line = '\t'.join([item.lower() for item in line])
    class1_out.append([line, 1])


# Write: data, path
helper.fwritejson(class0_out, out0)
helper.fwritejson(class1_out, out1)


