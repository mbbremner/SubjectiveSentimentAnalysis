# -*- coding: utf-8 -*-
# ====================================================================
# ---------------------------< WELCOME >------------------------------
# ====================================================================
""" This script tokenizes and processes the raw comment & tweet data
    Choose which dataset to process on line 32, run once for each data
    set"""

import re
import enchant
import time
import lstmhelper as helper

head = 'data/'
d = enchant.Dict("en_US")


bre_datapath = head + 'rawdata/breFullComments.txt'
bre_outpath = head + 'bre-comments-processed-400k.txt'
bre_dictpath = head + 'bre-dict-processed-400k.txt'

# youtubedatapath = head + 'rawdata/youtube_comments_raw.txt'
youtubedatapath = head + 'rawdata/AllYoutubeCommentsDatabase.txt'
youtubeoutpath = head + 'youtube-comments-processed-400k.txt'
youtubedictpath = head + 'youtube-dict-processed-400k.txt'

trumpdatapath = head + 'rawdata/tweetText.txt'
trumpoutpath = head + 'trump-tweets-processed.txt'
trumpdictpath = head + 'trump-dict-processed.txt'

useless_comment_path_trump = head + 'trump-useless-comments.txt'
useless_comment_path_youtube = head + 'youtube-useless-comments.txt'
useless_comment_path_bre = head + 'bre-useless-comments.txt'

bre_paths = (bre_datapath, bre_outpath, bre_dictpath, useless_comment_path_bre)
youtube_paths = (youtubedatapath, youtubeoutpath, youtubedictpath, useless_comment_path_youtube)
trumppaths = (trumpdatapath, trumpoutpath, trumpdictpath, useless_comment_path_trump)

# ------- < Choose from above  >----------
# datapath, outpath, dictpath, useless_comment_path = trumppaths
datapath, outpath, dictpath, useless_comment_path = bre_paths

# -------------------------------------------------------------------
# ---------------------------< Script >-------------------------------
# --------------------------------------------------------------------

helper.welcomebanner('Comment & Tweet Pre-Processing')


workingdata = helper.freadlist(datapath)           # Open tab delimited comment data
N = len(workingdata)-1                             # Make this smaller when testing code
print("    >> Total # of Comments: %d" % N)


# REGEX Splitting Pattern
faces = re.compile('[😄😃😀☺😊😀😊😀😃😃😄😃😊😊😊😊😊☺☺☺☺😊😊😊☺😉😉😍😘😚😗😞😞😒😔😔😁😁😳😳😛😛😝'
                   '😝😙😜😣😢😂😭😪😥😰😓😩😫😨😱😡😠😤😖😆😋😷😎😴😵😲😟😦😧😈👿😮😬😐😕😯😶😇😏😑]')
punctPatt = re.compile(r'0-9*\.0-9+|\#[a-zA-Z]+|(\'[a-z]){1,2}|[\s\"!$%&()*+,\-\/;<=>?@\[\]^_`{|}~]|([0-9]+:[0-9]{2})|([^\w\s,])')
arabicpattern = re.compile('([\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]){3,}')
quotations = re.compile('«‹»›„‚“‟‘‛”’"❛❜❟❝❞❮❯⹂〝〞〟＂')
chinesepat = re.compile('([\u2E80-\u4fFF]{2,})')
flowers = re.compile('[⚜-💐]')
emojirange = re.compile('[-󾓭]')
funnyface = re.compile('¯\\_(ツ)_/¯')
heartpattern = re.compile('<+3+')
linkpattern = re.compile('http[s]*\S+')
atpattern = re.compile('(@\S+)')
laugh_pat = re.compile('(\S[a-z0-9]*[bwah][bwha]+[a-z0-9]*)')
# linkpattern = re.compile('http[s]*\S+')
# 💯


startTime = time.monotonic()             # Start Time
level1lines = []                         # First level list (lines removed)
level2lines = []                         # Second level list (Remaining lines modified)

# --------------------------------------------------------
# ---------< (1) Filter Endesirable Sentences >-----------
# --------------------------------------------------------
# Here we filter sentences which are foreign, non-sensical
#  or may possibly break the regex


uselesscomments = []
counts = [0, 0, 0, 0]
start = time.monotonic()
for l, line in enumerate(workingdata[0:N]):
    # Remove words
    commentText = line.strip('\n').split('\t')[0]
    # commentText = line.strip('\n').split('\t')[0]
    if not re.search('[a-zA-Z0-9]', commentText) or re.search('([0-9]{50})', commentText):
        counts[0] += 1
        uselesscomments.append(commentText)
    elif re.search(arabicpattern, commentText):
        counts[0] += 1
        uselesscomments.append(commentText)
    elif re.search(chinesepat, commentText):
        uselesscomments.append(commentText)
        counts[0] += 1
    else:
        level1lines.append(commentText)

# I save the useless comments to study later
helper.fwritelist([item + '\n' for item in uselesscomments], useless_comment_path)

# --------------------------------------------------------
# -------------< (2) PRIMARY PREPROCESSING >--------------
# --------------------------------------------------------
print("    >> Substituting & Splitting: ")
start = time.monotonic()
for l, line in enumerate(level1lines[0:len(level1lines)-1]):

    initialText = line.strip('\n')
    subText = initialText.lower()                                        # Apply lower case here
    # -----< Check for substitution cases >-----
    # subText = re.sub('\b[aA]*[haHA]*(ha|HA)*', 'BIGBIGLAUGH', subText)
    subText = re.sub(atpattern, ' ', subText)                            # @Whoever
    subText = re.sub(linkpattern, ' ', subText)                          # Links
    subText = re.sub('(#[a-zA-Z0-9]*)', ' ', subText)                    # Hashtags
    subText = re.sub('([\.]{3,})', ' ... ', subText)                     # Elipsis
    subText = re.sub('([0-9]+:[0-9]{2})', '', subText)                   # Time Stamps
    subText = re.sub('([0-9]*\.*[0-9]{30,})', '<LONGNUMBER>', subText)   # Extra long numbers
    subText = re.sub('\bha+h[ha]*', 'bigbigbiglaugh', subText)           # Laughs
    subText = re.sub('\bHA+H[HAha]*', 'bigbigbiglaugh', subText)         # Other Lughs
    subText = re.sub('^[fF]+[uU]+[cC]+[kK]+$', 'fuck', subText)          # cuss word 1
    subText = re.sub('^[sS]+[hH]+[iI]+[tT]+$', 'shit', subText)          # cuss word 2
    subText = re.sub('^[sS]+[hH]+[iI]+[tT]+$', 'bitch', subText)          # cuss word 3
    subText = re.sub('^[lL]+[Oo]+[lL]+$', 'lol', subText)                # lol
    subText = re.sub('^[oO]+[mM]+[fF]*[gG]+$', 'omg', subText)
    subText = re.sub('^[pP]+[lL]+[zZsS]+$', 'pls', subText)
    subText = re.sub('‘|’', '\'', subText)
    subText = re.sub('“|”', '"', subText)
    # -----< End Check for substitution cases >-----

    # Apply Large Regex Pattern to substituted text
    splitText = re.split(punctPatt, subText)
    splitText = [item for item in splitText if item is not None and item != '']

    if not re.search( r'(\w)(\1{3,})', subText):
        level2lines.append([splitText,  subText, initialText])

    if l % 10000 == 0:
        print("      -- %d. %s  ... %5.2f seconds" % (l, splitText[0:20], time.monotonic() - start))

print("    >> Execution time: %5.3f" % (time.monotonic() - start))


helper.fwritelist([','.join(line[0]) + '\t' + str(line[1]) + '\t' + str(line[2]) + ' \n' for line in level2lines], outpath)


# --------------------------------------------------------
# ---------------< (3) Make Dictionaries >----------------
# --------------------------------------------------------
fullDict = {}
lvl2_split_text = [line[0] for line in level2lines]
for l, line in enumerate(lvl2_split_text[0:N]):
    helper.incriment_dictionary_by_line(fullDict, line)

helper.writedicttofile(fullDict, dictpath)

print("    >> Started With %d Comments \n    >> Then had %d Comments \n    >> Finished with %d Comments" %
      (len(workingdata), len(level1lines), len(level2lines)))


exit(0)


# --------------------------------------------------------------------
# ----------------------------< End >---------------------------------
# --------------------------------------------------------------------

