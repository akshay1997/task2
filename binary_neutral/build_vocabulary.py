# Builds vocabulary on the basis of the tweets and save it in "train/vocab.txt"
# This will be used to build the feature vector of each tweet.
from collections import Counter
import nltk, json, csv
import heuristics as h
from nltk.corpus import stopwords
from spellcheck import spellCheck
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
sia = SIA()
stop_words = set(stopwords.words('english'))
vader_dict={}
spch=spellCheck()

def map_emoji_to_rating(word):
    if word == '__EMOT_SMILEY' or word == '__EMOT_LOVE' or word == '__EMOT_WINK':
        return 2
    if word == '__EMOT_LAUGH':
        return 1
    if word == '__EMOT_FROWN':
        return 3
    if word == '__EMOT_CRY':
        return 4
    
def spellingcheck(tokens):
    rem=[]
    for item in tokens:
        item=item.lower()
        a=spch.correct(item)
        print item, a
        rem.append(a)
    return rem

def acronymapping(tokens):
    final=[]
    for item in tokens:
        if h.acronym_dict.has_key(item.lower()):
            final.extend(h.acronym_dict[item].split())
        else:
            final.extend(item.split())
    return final


def calc_adj(doc):
    global cntt
    tokens=doc.split()
    rem=[]
    rettok=[]
    for word in tokens:
        #word=word.lower()
#        if not word.isalpha():
#            continue
        if word == '__HNDL':
            pass
        elif word == '__URL':
            pass
        elif '__PUNC' in word:
            pass
        elif '__HASH' in word:
            pass
        elif word == '__NEG':
            pass
        elif '__EMOT' in word:
            res = map_emoji_to_rating(word)
            #rettok[res-1]+=1
        else:
            rem.append(word)
    #rem=acronymapping(rem)
    #rem = spellingcheck(rem)
    #print rem
    tagged = nltk.pos_tag(rem)
    for item in tagged:
        # extracting all the POS namely ADJ, ADV, VERBS, NOUNS
        if 'JJ' in item[1] or 'RB' in item[1] or 'VB' in item[1] or 'NN' in item[1]:
            rettok.append(item[0])
    cntt+=1
    return rettok

#-------------------------------------------------------------------------
cntt=0
vocab=Counter() # Initialised Vocab dictionary


with open('train/balanced.csv', 'rb') as inp:
    reader=csv.reader(inp)
    for row in reader:
        vocab.update(calc_adj(row[2]))
#print(vocab.most_common(100))
min_occurane = 5
# selecting all the words having occured at least 5 times
tokens = [k for k,c in vocab.iteritems() if c >= min_occurane]
with open('train/vocab.txt', 'wb') as op:
   op.write(json.dumps(tokens))
print len(tokens)
   
   

