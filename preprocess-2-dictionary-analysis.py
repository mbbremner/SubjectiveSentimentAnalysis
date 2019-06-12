# --------------------------------------------------------------------
# ---------------------------< WELCOME >------------------------------
# --------------------------------------------------------------------
""" This script is used to analyze the vocab & attempt to recover as many
    usable comments from the dataset while maintaining concise vocabulary.

    -A usable comment is a comment with zero unknown words.
    -Unknown words are words in the vocabulary that do not meet the frequency
     threshhold.
    - Loosening the threshhold allows for more usable comments but also
      expands the vocab with rare & ambiguous words & spellings.
    - The output will tell you how many sentences you recovered

    Summary: Analyze the dictionaries & sentences and, establish various dictionaries
    such as upper-case, lower-case, and restrained ( freq > X) dictionaries.
    Additionally, parse through the data-set and remove sentences which are
    are not fully composed of vocab items.
"""

# A. Imports
import lstmhelper as helper
import gluonnlp as nlp
import re


helper.welcomebanner(' Pre-Processing (2) ')

# B. If we choose, we may merge the wikitext vocab with the
# the vocab that we develop from the comments & tweets.
# For the moment this vocab is does not merge with ours,
# though it may
vocab = nlp.data.WikiText2(bos=None, eos='<eos>', skip_empty=False)
wiki2words = list(set([word.lower() for word in list(set(vocab))]))
print("\n    >> wikitext2 vocab length: %d" % (len(wiki2words)))
wiki2dict = {}
for word in wiki2words:
    wiki2dict[word] = 1

# --------------------------------------------------------------------
# --------------------------< Functions >-----------------------------
# --------------------------------------------------------------------


def analyze_words(tuple_list, pattern):
    """soxstring: don't forget your socks"""
    found = [word for word in tuple_list if re.search(pattern, word[0])]
    return found


def find_good_sents(adict, comment_list):
    """This function searches each comment and checks each word against
        the given dictionary.  If every word in the comment is in the dictionary,
        then it is a good sentence, otherwise bad.  We can use this to filter
        the data based on a vocab restriction"""

    goodlist = []
    badlist = []
    for s, sent in enumerate(comment_list):
        good = True
        for word in sent:
            if word.lower() not in adict:
                good = False
                badlist.append(sent)
                break
        if good:
            goodlist.append(sent)
    return goodlist, badlist


def combine_dicts(dict_a, dict_b):
    """Merge two dictionaries, combining counts"""

    tempdict = {}
    for item in dict_a.items():
        tempdict[item[0]] = item[1]
    for item in dict_b.items():
        if item[0] in tempdict:
            tempdict[item[0]] += item[1]
        else:
            tempdict[item[0]] = item[1]
    return tempdict


def print_every_x(x, inputdata):
    """A print function that puts X elements per line """
    # print(inputdata)
    print(range(int(len(inputdata)/x)))
    for i in range(int(len(inputdata)/x)):
        print(' '.join(inputdata[x*i:x*i+x]))
# --------------------------------------------------------------------
# ----------------------------< SCRIPT >------------------------------
# --------------------------------------------------------------------


freq_limit = 5   # Restrict vocab words by frequency

all_comments = [line.split('\t')[0].split(',') for line in helper.freadlist('data/youtube-comments-processed-400k.txt')]


dict0_path ='data/youtube-dict-processed-400k.txt'
dict1_path = 'data/bre-dict-processed-400k.txt'
# dict1_path = 'data/trump-dict-processed-400k.txt'

good_path = 'data/comments-good-bre-youtube-400k.txt'
bad_path = 'data/comments-bad-bre-youtube-400k.txt'
combined_d_path = 'data/dicts/combined-dict-lower-bre-youtube-f' + str(freq_limit)+'-400k.txt'


# Read in the dictionaries (non lower case form)
class1lines = helper.freadlist(dict1_path)
class1dict = helper.populatedict(class1lines, '\t')

youtube_lines = helper.freadlist(dict0_path)
ytdict = helper.populatedict(youtube_lines, '\t')


# 1. Sentences in the youtube comments containing only trump vocab words
#    I didn't really use this for anything, I was just curious how many of
#    the youtube comments matched Trump's vocabulary
# class1good = ['\t'.join(line) + ' \n' for line in find_good_sents(class1dict, all_comments)[0]]
# helper.fwritelist(class1good, 'data/trump-vocab400k-youtubegoodsents.txt')


# 2. Make lower case dictionaries
#    We are lower-casing everything because an upper
#    case dictioniary is way too large for fun training
ytdictlower = helper.dicttolower(ytdict)
class1dictlower = helper.dicttolower(class1dict)

# 3. Compute Dictionary Intersections
intersectionA = helper.dictintersection(ytdict, class1dict)
intersectionB = helper.dictintersection(ytdictlower, class1dictlower)

# 4. Set frequency limit on words &  build a constrained dictionary
youtube_dict_constrained = {key: ytdictlower[key] for key in ytdictlower if ytdictlower[key] > freq_limit}
class1_dict_constrained = {key: class1dictlower[key] for key in class1dictlower if class1dictlower[key] > freq_limit}

# 5. Combine Dictionaries
combined_dict = combine_dicts(youtube_dict_constrained, class1_dict_constrained)


# 6. Good & Bad sents are explained in he function "find_good_sents"
goodsents, badsents = {}, {}
goodsents['class1'], badsents['class1'] = find_good_sents(combined_dict, all_comments)
goodsents['youtube'], badsents['youtube'] = find_good_sents(youtube_dict_constrained, all_comments)


# Just some output verification
print("    >> Frequency Constraint: %d" % freq_limit)
# print("    >> Usable Sentences Trump: %d / %d\n" % (len(goodtrump), len(all_comments)))
print("    >> Youtube -- Upper -> Lower: %d -> %d" % (len(ytdict), len(ytdictlower)))
print("    >> Trump -- Upper -> Lower: %d -> %d" % (len(class1dict), len(class1dictlower)))
print("    >> Trump Youtube Intersection Upper / Lower: %d %d" % (len(intersectionA), len(intersectionB)))
print("    >> Shortened Lower: Youtube / Trump: %d,  %d" % (len(youtube_dict_constrained), len(class1_dict_constrained)))

print("\n    >> Combined Dict Size: %d " % len(combined_dict))
print("\n    >> Usable Sentences Youtube Only: %d / %d" % (len(goodsents['youtube']), len(all_comments)))
print("    >> Usable Sentences Combined Trump: %d / %d\n" % (len(goodsents['class1']), len(all_comments)))


# 7. Many forms of the vocabularies are written to a file here
# the '-short' versions have been truncated by the 'freq_limit' set
# many lines above.  Combined-lower-short is the one to use for
# NN training. Tinker with the 'freq_limit' value to see how vocab
# restrictions may effect the results.
# helper.writedicttofile(ytdictlower, head + 'dict-lower-youtube.txt')
# helper.writedicttofile(trumpdictlower, head + 'dict-lower-trump.txt')
# helper.writedicttofile(youtube_dict_constrained, head + 'dict-lower-youtube-short.txt')
# helper.writedicttofile(trump_dict_constrained, head + 'dict-lower-trump-short.txt')
helper.writedicttofile(combined_dict, combined_d_path)

# 8. Write Good / Bad comments to file.  The Good comments may be
# given to the next script, preprocess-3, for use in training and
# the bad comments may be studied for a compact way to include as
# many as possible with minimal inflation to the vocabulary.
helper.fwritelist(['\t'.join(item)+'\n' for item in goodsents['youtube']], good_path)
helper.fwritelist(['\t'.join(item)+'\n' for item in badsents['youtube']], bad_path)

exit(0)


# ====================================================================
# -----------------------------< End >--------------------------------
# ====================================================================


"""Everything down here is supplemental dictionary analysis that is incomplete"""



# len1sents = [item[0] for item in all_comments for i in range(20) if len(item) == i]
# print(len(all_comments))
# print(len(len1sents))

# # Compare wiki2dict to fully combined dict of bre trump & youtube
# specialkeys = [key for key in wiki2dict.keys() if key not in combinedDict]
# notsospecialkeys = [key for key in wiki2dict.keys() if key in combinedDict]
#
# helper.fwritelist([key + '\n' for key in specialkeys], 'processeddata/specialkeys.txt')
# print("    >> Wiki2 words not in / in data vocab: %d  %d" % (len(specialkeys), len(notsospecialkeys)))
#
# print("    >> Execution Time: %5.3f" % (time.monotonic()-start))
#
#
# f.close()
#
#
# badworddict = {}
# # Fetch comments by how many words need fixing to make it a usable comment
# words1count = [item for item in badsentlist if item[0] == 1]
#
# for item in words1count:
#     for word in item[1]:
#         word = word.lower()
#         if word not in badworddict:
#             badworddict[word] = 1
#         else:
#             badworddict[word] += 1
#
# badwordlist = [str(item[0]) + '\t' + str(item[1]) + '\n' for item in sorted(badworddict.items(), key=lambda x: x[1])]
# helper.fwritelist(badwordlist, 'processeddata/AAAbadworddict.txt')
#
# # Partitioned by counts
# many = [item[1] for item in badworddict.items() if item[1] > 1]
# few = [item[1] for item in badworddict.items() if item[1] == 1]
# # Length 1 keys
# len1keys = [item[0] for item in badworddict.items() if len(item[0]) == 1]
# len1counts = [item[1] for item in badworddict.items() if len(item[0]) == 1]
#
# print("    >> Total Sentences: %d ", len(words1count))
# print("    >> Len: %d  , Freq > 1 is %d" % (len(many), sum(many)))
# print("    >> Len: %d  , Freq == 1 is %d" % (len(few), sum(few)))
# print("    >> # of Len 1 items: %d" % len(len1keys))
# print("    >> # of sentences saved: %d" % sum(len1counts))
# # printeveryx(25, len1keys)


# --------------------------------------------------------------------
# ------------------------------< END >-------------------------------
# --------------------------------------------------------------------