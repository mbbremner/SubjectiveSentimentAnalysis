# -*- coding: utf-8 -*-
# ====================================================================
# ------< Principal Component Analysis with LSTM Features >-----------
# ====================================================================
"""
This is where I do PCA on the features ... or at least try to... :O
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import lstmhelper as H
import random
import time
import re
import math



def examine_ratios(components):
    ratios = []
    xlist = components[0]
    ylist = components[1]

    for i, item in enumerate(xlist):
        x, y = item, ylist[i]
        ratios.append([y/x, (x, y)])
        # print(" %5.4f / %5.4f  =  %5.4f" % (x, y, y/x))
    angles = []
    for r, ratio in enumerate(sorted(ratios)):
        # if r < 10 or r > len(ratios) - 10:
            # print(ratio)
            angle = np.arctan(ratio[1][1]/ratio[1][0]) * 90/np.pi - 90
            angles.append(angle)


    for angle in sorted(angles):
        print(angle)
    print('\n')

    return angles



def rotate_vector(components, angle_deg):


    rotated_components = []
    rotated_components.append([])
    rotated_components.append([])

    phi = angle_deg * np.pi/180
    for c, classlabel in enumerate(components):
        # rotated_components.append([])
        rotated_x = []
        rotated_y = []
        for i, item in enumerate(classlabel[0]):

            x, y = item, classlabel[1][i]
            xnew = x*np.cos(phi) - y*np.sin(phi)
            ynew = x*np.sin(phi) + y*np.cos(phi)
            rotated_x.append(xnew)
            rotated_y.append(ynew)
            # print("  Before: %5.4f %5.4f" % (x, y))
            # print("  After: %5.4f %5.4f" % (xnew, ynew))
        rotated_components[c] = [rotated_x, rotated_y]

    rotated_components = np.array(rotated_components)

    return rotated_components


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.clf()
plt.close()

# sigmoid = False
# activation = "Sigmoid"
activation = 'ReLU'
run = 'run1'
relu = False


random.seed(111)    # For noise generation on plots
num_components = 2  # Number of PCA components


# ====================================================================
# --------------------------< Functions >-----------------------------
# ====================================================================


def load_run_pca(featurepath, numcomponents):
    """Return PCA components from input data
        The lines are split according to the format that the
        features are saved in:  index \t label \t tabdelimited features"""
    lines = H.freadlist(featurepath)
    lines = [l.split('\t') for l in lines]

    data_id = np.transpose(np.int_([line[0] for line in lines]))
    data_lengths = np.transpose(np.int_([line[1] for line in lines]))
    predictions = np.transpose(np.float_([line[2] for line in lines]))
    data_labels = np.transpose(np.int_([line[3] for line in lines]))
    data_features = np.transpose(np.float_([line[4].split(',') for line in lines]))

    myPCA = PCA(n_components=numcomponents)
    myPCA.fit(data_features)
    return myPCA, data_labels, data_id, predictions, data_lengths


def prep_data(pcadata, labels, id_list):
    """Docstring"""

    pcapairs = [(item, labels[i], id_list[i]) for i, item in enumerate(np.transpose(pcadata.components_))]

    # for pair in pcapairs:
    #     print(pair)
    # Separate x,y for negative and positive case
    pair0 = np.transpose([item for item, label, id in pcapairs if label == 0]).tolist()
    id0 = np.transpose([id for item, label, id in pcapairs if label == 0]).tolist()
    pair1 = np.transpose([item for item, label, id in pcapairs if label == 1]).tolist()
    id1 = np.transpose([id for item, label, id in pcapairs if label == 1]).tolist()

    xydata = np.array([pair0, pair1])
    ids = np.array([id0, id1])

    return xydata, ids


def prep_normalized_data(pcadata, labels):
    """Condition the data for plotting"""
    noise = 0.1
    low, high = 1 - noise, 1 + noise
    a, b = 1.1, 1.0
    pcapairsnorm = [(random.uniform(low, high) * item/(np.linalg.norm(item)+10e-8), labels[i]) for i, item in enumerate(np.transpose(pcadata.components_))]
    pari1 = np.transpose([a*item for item, label in pcapairsnorm if label == 0]).tolist()
    pair2 = np.transpose([b*item for item, label in pcapairsnorm if label == 1]).tolist()
    xynormalized = np.array([pari1, pair2])
    return xynormalized


def addplot(fig, intuple, title, xylabels, colors):
    """docstring"""
    minpad, maxpad = 1.5, 1.1   # Padding

    # print(intuple)
    # print(len(intuple))
    # print(len(intuple[0]))
    negative = fig.scatter(intuple[0][0], intuple[0][1], color=colors[0], s=0.5, marker='o')
    positive = fig.scatter(intuple[1][0], intuple[1][1], color=colors[1], s=0.5, marker='o')



    xmin, xmax = min(min(min(intuple[0][0]), min(intuple[1][0])), -min(min(intuple[0][0]), min(intuple[1][0]))), max(max(intuple[0][0]), max(intuple[1][0]))
    ymin = min(-min(intuple[0][1]), -min(intuple[1][1]), min(intuple[0][1]), min(intuple[1][1]))
    # print(ymin)
    ymax = max(0.1*abs(ymin), max(max(intuple[0][1]), max(intuple[1][1])))

    fig.plot([0, 0], [-10, 10], 'k-', lw=1)     # x-axis
    fig.plot([-10, 10], [0, 0], 'k-', lw=1)     # y-axis

    fig.set_xlim(minpad*xmin, maxpad * xmax)
    fig.set_ylim(minpad*ymin, maxpad * ymax)
    fig.set_axisbelow(True)
    fig.grid(which='major')
    fig.set_title(title)
    fig.set_xlabel(xylabels[0])
    fig.set_ylabel(xylabels[1])
    fig.legend((negative, positive), ('Youtube', 'Trump'), prop={'size': 12}, loc='upper left', markerscale=6)


def addplot2(fig, intuple, title, xylabels, colors):
    """docstring"""
    minpad, maxpad = 1.5, 1.1   # Padding

    # print(intuple)
    # print(len(intuple))
    print("Addplot 2 Tuples : ")
    print(len(intuple[1]))
    print(len(intuple[0]))

    print(len(intuple[0][0]))
    print(len(intuple[0][1]))
    fig.scatter(intuple[0][0], intuple[0][1], color=colors[0], s=0.5, marker='o')
    # positive = fig.scatter(intuple[1][0], intuple[1][1], color=colors[1], s=0.5, marker='o')

    xmin, xmax = min(min(intuple[0][0]), -min(intuple[0][0])), max(intuple[0][0])
    ymin = min(-min(intuple[0][1]), min(intuple[0][1]))
    # # print(ymin)
    ymax = max(0.1*abs(ymin), max(max(intuple[0][0]), max(intuple[0][1])))

    fig.plot([0, 0], [-10, 10], 'k-', lw=1)     # x-axis
    fig.plot([-10, 10], [0, 0], 'k-', lw=1)     # y-axis

    fig.set_xlim(minpad*xmin, maxpad * xmax)
    fig.set_ylim(minpad*ymin, maxpad * ymax)
    fig.set_axisbelow(True)
    fig.grid(which='major')
    fig.set_title(title)
    fig.set_xlabel(xylabels[0])
    fig.set_ylabel(xylabels[1])
    # fig.legend((negative, positive), ('Youtube', 'Trump'), prop={'size': 12}, loc='upper left', markerscale=6)


def make_pca_quad_plot():
    """docstring"""
    colors = ['tab:blue', 'tab:red']
    fig1, f1_axes = plt.subplots(ncols=2, nrows=2, num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')

    # addplot(f1_axes, test_components, "Test Features", (None, "Principle Component # 2"), colors)
    addplot(f1_axes[1][0], test_components, "Test Features", ("Principle Component # 1", "Principle Component # 2"), colors)
    addplot(f1_axes[1][1], test_normalized, "Test Normalized", ("Principle Component # 1", None), colors)
    addplot(f1_axes[0][0], train_components, "Training Features", (None, "Principle Component # 2"), colors)
    addplot(f1_axes[0][1], train_normalized, "Training Normalized", (None, None), colors)

    # # Plot Centroids
    m = 'X'
    m_size = 18
    f1_axes[0][0].plot(train0c[0], train0c[1], color=colors[0], marker=m, markersize=m_size)
    f1_axes[0][0].plot(train1c[0], train1c[1], color=colors[1], marker=m, markersize=m_size)

    f1_axes[1][0].plot(test0c[0], test0c[1], color=colors[0], marker=m, markersize=m_size)
    f1_axes[1][0].plot(test1c[0], test1c[1], color=colors[1], marker=m, markersize=m_size)

    # Plot Centroid Radial Centroid Lines
    # Training Plot
    line_w = 1.5

    if relu:
        # ReLU Only
        plot_radial_centroid(f1_axes[0][1], q4_0c, colors[0], '--', line_w)
        plot_radial_centroid(f1_axes[0][1], q4_1c, colors[1], '--', line_w)
        plot_radial_centroid(f1_axes[0][1], q3_0c, colors[0], '--', line_w)
        plot_radial_centroid(f1_axes[0][1], q3_1c, colors[1], '--', line_w)

        plot_radial_centroid(f1_axes[1][1], q4_0c, colors[0], '--', line_w)
        plot_radial_centroid(f1_axes[1][1], q4_1c, colors[1], '--', line_w)
        plot_radial_centroid(f1_axes[1][1], q3_0c, colors[0], '--', line_w)
        plot_radial_centroid(f1_axes[1][1], q3_1c, colors[1], '--', line_w)

    plot_radial_centroid(f1_axes[0][1], c0train_norm, colors[0], '-', 2)
    plot_radial_centroid(f1_axes[0][1], c1train_norm, colors[1], '-', 2)

    # Testing Plot
    plot_radial_centroid(f1_axes[1][1], c0test_norm, colors[0], '-', 2)
    plot_radial_centroid(f1_axes[1][1], c1test_norm, colors[1], '-', 2)

    # 6. Touchups to plot figure
    fig1.suptitle(activation + " Features: Principle Components", fontsize=16, y=0.96)
    f1_axes[0][1].set_ylabel(None)
    f1_axes[1][1].set_ylabel(None)
    f1_axes[0][0].set_xlabel(None)
    f1_axes[0][1].set_xlabel(None)

    plt.show()
    plt.clf()
    plt.close()


def make_pca_quad_plot2():
    """docstring"""

    colors = ['tab:blue', 'tab:red']
    fig1, f1_axes = plt.subplots(ncols=2, nrows=1, num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')

    # addplot(f1_axes, test_components, "Test Features", (None, "Principle Component # 2"), colors)
    addplot2(f1_axes[0], leftover_components, "Unseen Comments: Raw Features", ("Principle Component # 1", "Principle Component # 2"), colors)
    addplot2(f1_axes[1], leftover_normalized, "Unseem Comments: Normalized ", ("Principle Component # 1", None), colors)
    # addplot(f1_axes[1], train_components[1], "Unseen Comments: Raw Features", ("Principle Component # 1", "Principle Component # 2"), colors)
    # addplot(f1_axes[0][0], train_components, "Training Features", (None, "Principle Component # 2"), colors)
    # addplot(f1_axes[0][1], train_normalized, "Training Normalized", (None, None), colors)

    # # Plot Centroids
    m = 'X'
    m_size = 18
    f1_axes[0].plot(leftc[0], leftc[1], color=colors[0], marker=m, markersize=m_size)
    # f1_axes[0].plot(train1c[0], train1c[1], color=colors[1], marker=m, markersize=m_size)
    #
    # f1_axes[1].plot(test0c[0], test0c[1], color=colors[0], marker=m, markersize=m_size)
    # f1_axes[1].plot(test1c[0], test1c[1], color=colors[1], marker=m, markersize=m_size)

    # Plot Centroid Radial Centroid Lines
    # Training Plot
    line_w = 1.5

    if relu:
        # ReLU Only
        plot_radial_centroid(f1_axes[0][1], q4_0c, colors[0], '--', line_w)
        plot_radial_centroid(f1_axes[0][1], q4_1c, colors[1], '--', line_w)
        plot_radial_centroid(f1_axes[0][1], q3_0c, colors[0], '--', line_w)
        plot_radial_centroid(f1_axes[0][1], q3_1c, colors[1], '--', line_w)

        plot_radial_centroid(f1_axes[1][1], q4_0c, colors[0], '--', line_w)
        plot_radial_centroid(f1_axes[1][1], q4_1c, colors[1], '--', line_w)
        plot_radial_centroid(f1_axes[1][1], q3_0c, colors[0], '--', line_w)
        plot_radial_centroid(f1_axes[1][1], q3_1c, colors[1], '--', line_w)

    print(leftc_norm)
    plot_radial_centroid(f1_axes[1], [leftc_norm[0], leftc_norm[1]], colors[0], '-', 2)

    # phi = -155 * np.pi / 180
    # xnew = train1c[0] * np.cos(phi) - train1c[1] * np.sin(phi)
    # ynew = train1c[0] * np.sin(phi) + train1c[1] * np.cos(phi)
    plot_radial_centroid(f1_axes[1], train1c, colors[1], '-', 2)
    # plot_radial_centroid(f1_axes[0][1], c0train_norm, colors[0], '-', 2)
    # plot_radial_centroid(f1_axes[0][1], c1train_norm, colors[1], '-', 2)
    #
    # # Testing Plot
    # plot_radial_centroid(f1_axes[1][1], c0test_norm, colors[0], '-', 2)
    # plot_radial_centroid(f1_axes[1][1], c1test_norm, colors[1], '-', 2)
    #
    # # 6. Touchups to plot figure
    # fig1.suptitle(activation + " Features: Principle Components", fontsize=16, y=0.96)
    # f1_axes[0][1].set_ylabel(None)
    # f1_axes[1][1].set_ylabel(None)
    # f1_axes[0][0].set_xlabel(None)
    # f1_axes[0][1].set_xlabel(None)

    plt.show()
    plt.clf()
    plt.close()


def plot_radial_centroid(input_plot, centroid, col, style, w):
    """Docstring 4 u"""
    # print(centroid)
    centroid = centroid / np.linalg.norm(centroid)
    input_plot.plot([0, 1.2 * centroid[0]], [0, 1.2 * centroid[1]], col, lw=w, linestyle=style)


def compare_features_to_centroid(input_pairs, centroid):
    """There, a docstring, are you happy now?"""
    similarities = []

    for p, pair in enumerate(input_pairs[0]):
        x = input_pairs[0][p]
        y = input_pairs[1][p]
        similarity = H.cosine_similarity([x,y], centroid)
        similarities.append(similarity)

    return similarities


def separate_by_class(input_data):
    """ditto"""
    class0 = [item for item in input_data if int(item[1]) == 0]
    class1 = [item for item in input_data if int(item[1]) == 1]

    return class0, class1


def select_features_by_quadrant(data0, data1):
    """Docstring 4 u"""
    q3_1 = np.array([t / np.linalg.norm(t) for t in data1.T if t[0] < 0 and t[1] < 0])
    q4_1 = np.array([t / np.linalg.norm(t) for t in data1.T if t[0] > 0 and t[1] < 0])

    q3_0 = np.array([t / np.linalg.norm(t) for t in data0.T if t[0] < 0 and t[1] < 0])
    q4_0 = np.array([t / np.linalg.norm(t) for t in data0.T if t[0] > 0 and t[1] < 0])

    return q3_0, q3_1, q4_0, q4_1


def compute_centroid_transpose(data):
    """Docstring 4 u"""
    return [np.mean(data.T[0]), np.mean(data.T[1])]


def compute_centroid(data):
    """docstring"""
    return [np.mean(data[0]), np.mean(data[1])]


# head = 'data/features/sigmoid/features_run1_'
# head = 'data/features/sigmoid'
# leftover_path = head + 'sgd_leftovers.txt'

# ====================================================================
# ---------------------------< Script >-------------------------------
# ====================================================================
H.welcomebanner(' LSTM Principle Component Analysis')

# Comment Text - Path
# h = 'data/jsondata/bre/run2/'
# head = 'data/features/relu/bre-8-epoch-'
# head = 'data/features/sigmoid/bre-'
# train_path, test_path = head+'sigmoid-train.txt', head+'sigmoid-test.txt'
# # train_path, test_path = head+'relu-train.txt', head+'relu-test.txt'
# leftover_path = head + 'youtube-leftover.txt'
# train_file, test_file, leftover_file = h + 'breTrain.json', h + 'breTest.json', h + 'youtube_leftover_comments.json'

# Feature - Path
h = 'data/jsondata/trump/run1/'

train_file, test_file, leftover_file = h + 'trumpTRAIN.json', h + 'trumpTEST.json', h + 'leftovers.json'
# head = 'data/features/sigmoid/trump-run2-sigmoid-'
# head = 'data/features/relu/trump-run2-relu-'
# head = 'data/features/sigmoid/features_run1_'
head = 'data/features/relu/features_run1_relu_4ep_'
# head = 'data/features/sigmoid/trump-run2-sigmoid-'
# train_path, test_path = head+'trumpTRAIN.txt', head+'trumpTEST.txt'
train_path, test_path = head+'trumpTRAIN.txt', head+'trumpTRAIN.txt'
leftover_path = head + 'leftovers.txt'


# Comment Text - Load
all_train_text, all_text_data = H.fopenjson(train_file), H.fopenjson(test_file)
leftover_comments = H.fopenjson(leftover_file)

class0_train_text, class1_train_text = separate_by_class(all_train_text)
class0_test_text, class1_test_text = separate_by_class(all_text_data)


# ---------------------------------------------
# 1. ----------< Load Data & run PCA >---------
# ---------------------------------------------
train_pca, trainlabs, train_id, train_predictions, train_lens = load_run_pca(train_path, numcomponents=2)
test_pca, testlabs, test_id, test_predictions, test_lens = load_run_pca(test_path, numcomponents=2)
leftover_pca, leftover_labs, leftover_id, leftover_predictions, leftover_lens = load_run_pca(leftover_path, numcomponents=2)
# ---------------------------------------------
# 2. -------< Format Data for Plotting >-------
# ---------------------------------------------
# Unnormalized
test_components, test_ids = prep_data(test_pca, testlabs, test_id)
train_components, train_ids = prep_data(train_pca, trainlabs, train_id)
leftover_components, left_ids = prep_data(leftover_pca, leftover_labs, leftover_id)

# Normalized
test_normalized = prep_normalized_data(test_pca, testlabs)
train_normalized = prep_normalized_data(train_pca, trainlabs)
leftover_normalized = prep_normalized_data(leftover_pca, leftover_labs)

print(train_components[0].shape)
# r1 = examine_ratios(train_components[0])
# r2 = examine_ratios(train_components[1])
# r3 = examine_ratios(test_components[0])
# r4 = examine_ratios(test_components[1])


# print(min(min(r1), min(r2)))
# # v = min(min(r3), min(r4))*1.0
# v = max(max(r3), max(r4))*1.0

# exit(0)
# exit()
# train_components = rotate_vector(train_components, 135)
# train_normalized = rotate_vector(train_normalized, 135)
# # train_components = rotate_vector(test_components, 135.95) 130
# test_components = rotate_vector(test_components, 135)
# test_normalized = rotate_vector(test_normalized, 135)

# test_components = rotate_vector(test_components, 135)
# leftover_normalized = rotate_vector(leftover_normalized, 135)
# exit(0)
# ---------------------------------------------
# 3. -----------< Separate by Class >----------
# ---------------------------------------------
# a. Training
train0, train1 = train_components[0], train_components[1]
train_ids_0, train_ids_1 = train_ids[0], train_ids[1]
# b. Testing
test0, test1 = test_components[0], test_components[1]
test_ids_0, test_ids_1 = test_ids[0], test_ids[1]
# b. Leftover
left = leftover_components[0]
print('left')
print(len(left[0]))
print(len(left[1]))
# ---------------------------------------------
# 4. ----------< Compute Centroids >-----------
# ---------------------------------------------
# Regular Centroids
train0c, train1c = compute_centroid(train0), compute_centroid(train1)
test0c, test1c = compute_centroid(test0), compute_centroid(test1)
leftc = compute_centroid(left)
print(leftc)
# Normalized Centroids
c0train_norm, c1train_norm = train0c/np.linalg.norm(train0c), train1c/np.linalg.norm(train1c)
c0test_norm, c1test_norm = test0c/np.linalg.norm(test0c), test1c/np.linalg.norm(test1c)
leftc_norm = leftc/np.linalg.norm(leftc)

print(leftc_norm)

# Leftover Centroids


# ReLU quadrant specific centroids
if relu:
    # Training Centroids by quadrant
    q3_0, q3_1, q4_0, q4_1 = select_features_by_quadrant(train0, train1)
    q3_0c, q3_1c, q4_0c, q4_1c = compute_centroid(q3_0.T), compute_centroid(q3_1.T), compute_centroid(q4_0.T), compute_centroid(q4_1.T)
    # Testing Centroids by quadrant
    q3_0_test, q3_1_test, q4_0_test, q4_1_test = select_features_by_quadrant(test0, test1)
    q3_0c_test, q3_1c_test, q4_0c_test, q4_1c_test = compute_centroid(q3_0_test.T), compute_centroid(q3_1_test.T), compute_centroid(q4_0_test.T), compute_centroid(q4_1_test.T)

    print("Training: ")
    print("\tQ3 Centroids: %5.3f,%5.3f    %5.3f,%5.3f " % (q3_0c[0], q3_0c[1], q3_1c[0], q3_1c[1]))
    print("\tQ4 Centroids: %5.3f,%5.3f    %5.3f,%5.3f " % (q4_0c[0], q4_0c[1], q4_1c[0], q4_1c[1]))
    print("Testing: ")
    print("\tQ3 Centroids: %5.3f,%5.3f    %5.3f,%5.3f " % (q3_0c_test[0], q3_0c_test[1], q3_1c_test[0], q3_1c_test[1]))
    print("\tQ4 Centroids: %5.3f,%5.3f    %5.3f,%5.3f " % (q4_0c_test[0], q4_0c_test[1], q4_1c_test[0], q4_1c_test[1]))
    print("Youtube Third Quadrant: %d" % len(q3_0))
    print("Youtube Fourth Quadrant: %d" % len(q4_0))
    print("Trump Third Quadrant: %d" % len(q3_1))
    print("Trump Fourth Quadrant: %d" % len(q4_1))


# ---------------------------------------------
# 5. -------------< PCA Plots >----------------
# ---------------------------------------------
plt.clf()
plt.close()
# make_pca_quad_plot()
make_pca_quad_plot2()


# ---------------------------------------------
# 6. --------------< Analysis >----------------
# ---------------------------------------------
#
# A. << COSINE SIMILARITY >>
# Compare first parameter (list of vectors) to second (centroid vector)
features = 'test0'
features = 'leftovers'
input = leftover_normalized[0]
centroid = train1c
phi = -145 * np.pi / 180
centroid = (train1c[0] * np.cos(phi) - train1c[1] * np.sin(phi), train1c[0] * np.sin(phi) + train1c[1] * np.cos(phi))

text_data = leftover_comments

similarities = compare_features_to_centroid(input, centroid)
print(len(similarities))
similarity_tuples = [(leftover_id[i], similarities[i], text_data[i]) for i, item in enumerate(similarities)]

# Make a Ranked List of Comments
similar_comments = sorted(similarity_tuples, key=lambda b: b[1], reverse=True)
cosine_sim_sorted = [str(item[0]) + '\t' + str(item[1]) + '\t' + re.sub('\\t', ' ', item[2][0]) + '\n' for item in similar_comments]

H.fwritelist(cosine_sim_sorted, 'data/rankedsents/' + run + activation + features + '-cosinesimilarity.txt')
#
# # B.  << SIGMOID PREDICTION >>
# predictions = []
# tup = ()
# data = test_predictions
# for p, prediction in enumerate(data):
#
#     id_tag = train_id[p]
#     text = re.sub('\t', ' ', all_train_text[p][0])
#     lab = trainlabs[p]
#     leng = train_lens[p]
#
#     tup = (id_tag, leng, lab, prediction, text)
#     predictions.append(tup)
#
# preds_0 = sorted([item for item in [a for a in predictions if a[2] == 0]], key=lambda z: z[3], reverse=True)
# preds_1 = sorted([item for item in [a for a in predictions if a[2] == 1]], key=lambda z: z[3], reverse=True)


# # Write predictions sorted with sentences to file
# H.fwritelist(preds_0, 'data/rankedsents/' + run + activation + 'preds0')
# H.fwritelist(preds_1, 'data/rankedsents/' + run + activation + 'preds1')
#
#


# ====================================================================
# -----------------------------< End >--------------------------------
# ====================================================================


#
# # DBSCAN CLUSTERING (NOT IMPLIMENTING THIS ATM)
# # Best to simply compute a simple metric with
# # this volume of data
# print(len(test_components))
# print(len(train_components[0]))
# print(len(test_components[0]))
#
# components = train_normalized
#
# # components = [item / np.linalg.norm(item) for item in components[0]]
#
# negative_components = np.array([item / np.linalg.norm(item) for item in components[0]])
# positive_components = np.array([item / np.linalg.norm(item) for item in components[1]])
# print(len(negative_components))
# print(len(negative_components.T))
# # [x,y for item in a,b]
# #
#
# # db_input = negative_components.T[0:10000]
# # db_input = negative_components.T
# db_input_p = positive_components.T
# db_input_n = negative_components.T
#
#
# print(" Average vector Length: %f" % np.mean([np.linalg.norm(item) for item in db_input_p][0]))
# print(" Average vector Length: %f" % max([np.linalg.norm(item) for item in db_input_p]))
# print(" Average vector Length: %f" % min([np.linalg.norm(item) for item in db_input_p]))
#
#
# # for item in db_input:
# #     print(np.linalg.norm(item))
#
# print("  > Clustering ...")
# s = time.monotonic()
# clustering_p = DBSCAN(eps=0.0003, min_samples=150, algorithm='ball_tree').fit_predict(db_input_p)
# print(" Clustering Time:  %f" % (time.monotonic() - s))
# clustering_n = DBSCAN(eps=0.0003, min_samples=150, algorithm='ball_tree').fit_predict(db_input_n)
# print(" Clustering Time:  %f" % (time.monotonic() - s))
# # print(clustering)
#
# for i, item in enumerate(clustering_p):
#     if item == 1:
#         clustering_p[i] = -1
#
#
# print("Cluster Sizes:")
# print(len([item for item in clustering_p if item == 1]))
# print(len([item for item in clustering_p if item == 0]))
# print(len(clustering_p))
# print(len(clustering_n))
# print(set(clustering_p))
# print(set(clustering_n))
#
# fig2, f2_axes = plt.subplots(ncols=2, nrows=1, num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
# # cluster_plot(f2_axes[0], db_input_p, "Test Features", (None, "Principle Component # 2"))
# # cluster_plot(f2_axes[1], db_input_n, "Test Features", (None, "Principle Component # 2"))
#
# f2_axes[0].scatter(db_input_p.T[0], db_input_p.T[1], c=clustering_p, s=6, marker='x')
# f2_axes[1].scatter(db_input_n.T[0], db_input_n.T[1], c=clustering_n, s=6, marker='x')
#
# plt.show()


# ---------------- Word cloud stuffs ------------------
# plt.close()
# fig1, f1_axes = plt.subplots(ncols=2, nrows=1, num=None, figsize=(14, 8), dpi=80, facecolor='w', edgecolor='k')
#
# def make_word_cloud(input_data, ax, num_words):
#     alltext = ' '.join([re.sub('\\t', ' ', item[0]) for item in input_data])
#     wordcloud = WordCloud(max_font_size=80, relative_scaling=0.5, max_words=num_words,
#                           background_color="white", width=400, height=400).generate(alltext)
#     ax.imshow(wordcloud, interpolation='bilinear')
#     ax.axis("off")
#     # plt.axis("off")
#
# youtube_cloud_words = [i for i in train_text if i[1] == 0]
# trump_cloud_words = [i for i in train_text if i[1] == 1]
#
# make_word_cloud(youtube_cloud_words, f1_axes[0], 100)
# make_word_cloud(trump_cloud_words, f1_axes[1], 100)
# plt.show()


# ------------------- Length Plots --------------------
# mean = 0
# for i, item in enumerate(train_preds_0[0:100] + train_preds_0[len(train_preds_0)-100:]):
#     mean = (mean*i + item[1])/(i+1)
#     print(item[1], mean)
#
#     if i == 100:
#         print('\n')
#     # print("  %d %d %d  %5.8f  >>  %s" % item)
#
# mean = 0
# pred_mean = 0
# preds = train_preds_1
# modval = 1000
# for i, item in enumerate(train_preds_1):
#     k = i % modval
#     mean = (mean*k + item[1])/(k+1)
#     pred_mean = (pred_mean*k + item[3])/(k+1)
#     if i % modval == 99:
#         print("%d %5.2f  %5.8f" % (i, mean, pred_mean))
#         mean, pred_mean = 0, 0
    # print("  %d %d %d  %5.8f  >>  %s" % item)


# train_pca, trainlabs, train_id, train_predictions, train_lens

# print(len(train_components[0][0]))
# print(train_components[0].shape)
# print(train_components[1].shape)
# print(len(train_lens))
# print(len(trainlabs))
# len_limit = 10
# train_components = train_pca.components_.T
# short_comments = np.array([item for i, item in enumerate(train_components) if int(train_lens[i]) < len_limit and trainlabs[i] == 0])
# long_comments = np.array([item for i, item in enumerate(train_components) if int(train_lens[i]) >= len_limit and trainlabs[i] == 0])
#

# figa, fa_axes = plt.subplots(ncols=1, nrows=1, num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
# print(short_comments[0].shape)
#
# fa_axes.scatter(short_comments.T[0], short_comments.T[1], c='k', s=2)
# fa_axes.scatter(long_comments.T[0], long_comments.T[1], c='g', s=2)
# plt.show()

# print(len(short_comments))
# print(len(long_comments))
# exit(0)

