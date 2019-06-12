# -*- coding: utf-8 -*-
# --------------------------------------------------------------------
# ----------------< LSTM Project Helper Functions >-------------------
# --------------------------------------------------------------------
""" An aggregation of helpful functions used by several files"""

# Can't seem to remember this one so I keep it  here: vocab.idx_to_token

import io
import re
import numpy as np
import json
import argparse
from mxnet import gluon
import multiprocessing as mp
import time
import gluonnlp as nlp
# --------------------------------------------------------------------
# --------------------------< Functions >-----------------------------
# --------------------------------------------------------------------


def welcomebanner(title):
    """Docstring"""
    N = 80
    first = N - len(title)
    second = int(first/2)
    third = N - len(title) - second
    print('\n\n  ' + '=' * N)
    print('  ' + '-' * second + title + '-' * third)
    print('  ' + '=' * N)


def cosine_similarity(v1, v2):
    """Docstring"""
    # Handle for zero len vectors
    if np.linalg.norm(v2) == 0:
        similarity_value = 0.1111
    elif np.linalg.norm(v1) == 0:
        similarity_value = 0.1111
    # Else ...
    else:
        top = np.dot(v1, v2)
        bottom = np.linalg.norm(v1) * np.linalg.norm(v2)
        similarity_value = top / bottom

    return similarity_value


def dictintersection(dictA, dictB):
    """ Given two dicts, returns a list of common keys """
    interkeys = []
    for key in dictA.keys():
        if key in dictB:
            interkeys.append(key)

    return interkeys


# 1.-------------- Training & Feature Extraction ---------------
def lstm_argparser():
    """Standard argparser for LSTM Training & Feature Extraction"""
    ap = argparse.ArgumentParser()
    # Feature Extraction & Training
    ap.add_argument('-epochs', nargs=1, type=int, default=[1])
    ap.add_argument('-featurepath', type=str, nargs=1, default=['default_features.txt'])
    ap.add_argument('-jsonpath', type=str, nargs=1, default=['default.json'])
    ap.add_argument('-modelname', type=str, nargs=1, default=['default_model'])
    ap.add_argument('-acttype', type=str, nargs=1, default=['sigmoid'])


    # For Training
    ap.add_argument('-trainpath', type=str, nargs=1, default=['default_model'])
    ap.add_argument('-testpath', type=str, nargs=1, default=['default_model'])
    ap.add_argument('-dropout', type=float, nargs=1, default=[0.1])
    ap.add_argument('-learningrate', nargs=1, type=float, default=[0.005])
    ap.add_argument('-batchsize', type=int, nargs=1, default=[96])
    ap.add_argument('-vocabpath', type=str, nargs=1, default=['data/combinedDictionary.txt'])
    return ap



def preprocess_with_spacy(x):
    """This is just here in case we want to use spacy tokenizer again"""
    data, label = x
    # label = int(label > 5)
    label = int(label)
    # A token index or a list of token indices is
    # returned according to the vocabulary.
    # print("Tokenizer at Data: ")
    # print(tokenizer(data))
    # data = vocab[length_clip(tokenizer(data))]
    # data = vocab[length_clip(data.split('\t'))]
    return data, label


def loadvocabdictionary(path):
    """key \t value pairs read into a list & copied into a dictionary
       which is returned"""
    f = io.open(path, 'r')
    lines = f.readlines()
    f.close()

    tempDict = {}
    delim = '\t'
    for line in lines:
        word = line.split(delim)[0]
        count = int(line.split(delim)[1])
        tempDict[word] = count
    return tempDict


def fopenjson(inpath):
    """Docstring"""
    with open(inpath) as json_file:
        dataset = json.load(json_file)
        for i, item in enumerate(dataset): dataset[i][0] = item[0].strip('\n')
        json_file.close()
    return dataset


def fwritejson(out_data, out_path):
    """Docstring"""
    with open(out_path, 'w') as json_file:
        json.dump(out_data, json_file)
        json_file.close()

# 2. -------- Vocab analysis & Sentence Preprocessing ----------


def incriment_dictionary_by_word(some_dict, some_word):
    """docstring bitches"""
    if some_word.lower() not in some_dict:
        some_dict[some_word.lower()] = 1
    else:
        some_dict[some_word.lower()] += 1


def incriment_dictionary_by_line(some_dict, list_of_words):
    """docstring bitches"""
    for word in list_of_words:
        if word.lower() not in some_dict:
            some_dict[word.lower()] = 1
        else:
            some_dict[word.lower()] += 1


def incriment_dictionary_by_whole_list(list_of_list_of_words):
    """docstring bitches"""
    some_dict = {}
    for line in list_of_list_of_words:
        for word in line:
            if word.lower() not in some_dict:
                some_dict[word.lower()] = 1
            else:
                some_dict[word.lower()] += 1
    return some_dict


def showmesentenceswith(pattern, data):
    """ Display Sentences according to a regex pattern
     The sentences must be in the form of a single unbroken string"""
    matches = []
    for line in data:
        if re.search(pattern, line):
            matches.append(line)

    return matches


def writedicttofile(inputdict, filepath):
    """ Write a dictionary to file as tab delimited pairs
    Input is a dictionary obj and outpath"""

    inputsorted = [item for item in sorted(inputdict.items(), key=lambda x: x[1], reverse=True)]
    f = io.open(filepath, 'w', encoding='utf-8')
    for item in inputsorted:
        f.write(str(item[0]) + '\t' + str(item[1]) + ' \n')
    f.close()


# Write a list to file
def fwritelist(inlist, savepath):
    """ Basic write a list of strings to file"""
    f = io.open(savepath, 'w', encoding='utf-8')
    for line in inlist:
        f.write(line)
    f.close()

    print("    >> Saved to: %s" % savepath)


# Read lines to  a list
def freadlist(inpath):
    """ Read lines to  a list """
    f = io.open(inpath, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    return lines


def populatedict(pairlist, delim):
    """ Input is a list of strings """
    tempdict = {}
    for line in pairlist:
        word, count = line.split(delim)
        tempdict[word] = int(count)
    return tempdict


def populatedictnodelim(pairlist):
    """ Input is a list of tuples """
    tempdict = {}
    for word, count in pairlist:
        tempdict[word] = int(count)
    return tempdict


def dicttolower(inputdict):
    """ Make all keys lower and recombine into a new dictionary """
    lowerdict = {}
    for key, count in inputdict.items():
        if key.lower() not in lowerdict:
            lowerdict[key.lower()] = count
        else:
            lowerdict[key.lower()] += count

    return lowerdict



