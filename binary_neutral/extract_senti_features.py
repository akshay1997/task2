#Calculates the Senti-features tweet wise and store in "train/senti.csv"
import string, nltk, json, csv
import heuristics as h
from nltk.corpus import stopwords, sentiwordnet as swn, wordnet as wn
from spellcheck import spellCheck
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
sia = SIA()
stop_words = set(stopwords.words('english'))
vader_dict={}
#s = SentiWordNetCorpusReader("../nltk_data/corpora/sentiwordnet", ["SentiWordNet_3.0.0_20130122.txt"])
spch=spellCheck()
#example processed tweet ['__HNDL', '__URL', 'Aww', 'thats', 'a', 'bummer',
# 'You', 'shoulda', 'got', 'David', 'Carr', 'of', 'Third', 'Day', 'to', 
# 'do', 'it', '__EMOT_WINK']

def map_emoji_to_rating(word):
    if word == '__EMOT_SMILEY' or word == '__EMOT_LOVE' or word == '__EMOT_WINK':
        return 2
    if word == '__EMOT_LAUGH':
        return 1
    if word == '__EMOT_FROWN':
        return 3
    if word == '__EMOT_CRY':
        return 4
# Sentiment Vader Corpus for word polarity
def vader(word):
    try:
        return vader_dict[word]
    except KeyError:
        try:
            return sia.polarity_scores(word)['compound']
        except KeyError:
            return False
# sentiwornet corpus for word polarity
def sentiword(word, tagged):
    for item in tagged:
        if item[0] == word:
            tag=item[1]
    sen_sets=wn.synsets(word,pos=h.POS_LIST.get(tag))
    #If word not found in the sentiwordnet corpus, search for the synonym of the 
    # word in the sentiwordnet corpus and if found assign that polarity to
    # the original word.
    if not sen_sets:
        return 0.0
    try:
        a = swn.senti_synset(sen_sets[0].name())
    except:
        return 0.0
    pos,neg = a.pos_score(),a.neg_score()
    if pos>=neg:
        return a.pos_score()
    else:
        return -a.neg_score()


# helper function for finding the polarity of the passed tokens and their POS
def findpolarity(rem, tagged):
    polarity={}
    for word in rem:
        if h.polarity_list.has_key(word):
            # if not found search the polarity in sentiStrength corpus
            polarity[word]=h.polarity_list[word]
        elif h.polar_lookup.has_key(word):
            # finding in the wordStrength corpus
            polarity[word]=h.polar_lookup[word]
        elif vader(word): # sentiment Vader Corpus
            vader_dict[word]=vader(word)
            polarity[word]=vader_dict[word]
        try:
            # if polarity still now found, search using sentiwordnet
            if polarity[word] == 0.0:
                polarity[word]=sentiword(word, tagged)
        except KeyError:
            polarity[word]=sentiword(word, tagged)
    return polarity


def feature_extraction(tokens, tagged, polar, l1):
    # to extract the count of Polar and Non-Polar POS features
    # Polar-> having non-zero polarity (+ve or -ve)
    # Non Polar-> having zero polarity
    global cntt
    pos_dict={}
    for item in tagged:
        pos_dict[item[0]]=item[1]
    for item in tokens:
        #print pos_dict[item], polar[item]
        if 'NN' in pos_dict[item]:
            # extract count of positive, negative and neutral NOUNS.
            if polar[item] > 0.0:
                l[cntt][l1]+=1.0
            elif polar[item] < 0.0:
                l[cntt][l1+1]+=1.0
            else:
                l[cntt][l1+2]+=1.0
        # Similarly for Adj, Adv & Verbs and append them to the sentifeature vector.
        elif 'JJ' in pos_dict[item]:
            if polar[item] > 0.0:
                l[cntt][l1+3]+=1.0
            elif polar[item] < 0.0:
                l[cntt][l1+4]+=1.0
            else:
                l[cntt][l1+5]+=1.0
        elif 'RB' in pos_dict[item]:
            if polar[item] > 0.0:
                l[cntt][l1+6]+=1.0
            elif polar[item] < 0.0:
                l[cntt][l1+7]+=1.0
            else:
                l[cntt][l1+8]+=1.0
        elif 'VB' in pos_dict[item]:
            if polar[item] > 0.0:
                l[cntt][l1+9]+=1.0
            elif polar[item] < 0.0:
                l[cntt][l1+10]+=1.0
            else:
                l[cntt][l1+11]+=1.0
        else:
            if polar[item] > 0.0:
                l[cntt][l1+12]+=1.0
            else:
                l[cntt][l1+13]+=1.0

# Summing up the polarity of nouns, adj, adv, verbs in each tweet
def feature_sum(tokens, tagged, polar, l2):
    nn=0.0 # summation of polarities of all nouns in a tweet
    jj=0.0
    rb=0.0
    vb=0.0
    total=0.0 # summation of above 4
    global cntt
    pos_dict={}
    for item in tagged:
        pos_dict[item[0]]=item[1]
    for item in tokens:
        polar[item]=float(polar[item])
        if 'NN' in pos_dict[item]:
            nn=nn+polar[item]
        elif 'JJ' in pos_dict[item]:
            jj=jj+polar[item]
        elif 'RB' in pos_dict[item]:
            rb=rb+polar[item]
        elif 'VB' in pos_dict[item]:
            vb=vb+polar[item]
        else:
            total=total+polar[item]
    total=total+nn+jj+rb+vb
    l[cntt][l2]+=nn
    l[cntt][l2+1]+=jj
    l[cntt][l2+2]+=vb
    l[cntt][l2+3]+=rb
    l[cntt][l2+4]+=total # appended to the senti feature vector
    
def spellingcheck(tokens):
    rem=[]
    for item in tokens:
        item=item.lower()
        #a=spch.correct(item)
        print item, #a
        #rem.append(a)
    return rem

def acronymapping(tokens):
    final=[]
    for item in tokens:
        if h.acronym_dict.has_key(item.lower()):
            final.extend(h.acronym_dict[item].split())
        else:
            final.extend(item.split())
    return final

def process_hashtags(word):
    # determine whether the hashtag has positive, negative or neutral sentiment.
    str="__HASH_"
    word=string.replace(word, str, "")
    word=word.lower()
    word=word.split()
    tagged=nltk.pos_tag(word)
    polar=findpolarity(word, tagged)
    return polar

def clean_doc(doc):
    # computes the sentifeatures of each tweet
    global cntt
    tokens=doc.split()
    rem=[]
    polar={}
    for word in tokens:
        #if not word.isalpha():
            #continue
        # extracting the count of handles, urls, punctuations 
        if word == '__PUNC_EXCL':
            l[cntt][11]+=1.0 # checking for the presence of "!"
        elif word == '__HNDL':
            l[cntt][0]+=1.0
        elif word == '__URL':
            l[cntt][2]+=1.0
        elif '__PUNC' in word:
            l[cntt][1]+=1.0
        elif '__HASH' in word:
            val=process_hashtags(word) # detecting the sentiment of hashtag
            if val > 0.0:
                l[cntt][3]+=1.0
            elif val < 0.0:
                l[cntt][4]+=1.0
            else:
                l[cntt][5]+=1.0
        elif word == '__NEG':
            l[cntt][6]+=1.0 # replacing negation words with a __NEG tag
        elif '__EMOT' in word:
            res = map_emoji_to_rating(word)
            l[cntt][res+6]+=1.0 # mapping emoticons to ratings on a scale of 
            # 1-4
        else:
            # appending all other words
            rem.append(word.lower()) 
    #rem=acronymapping(rem)  # uncomment to do acronym mapping eg:
    # lol to laugh out loud
    
    #rem = spellingcheck(rem) # uncomment to implement spelling check 
    # corrects wrong words' spellings
    
    tagged = nltk.pos_tag(rem) # Parts Of Speech Tagging
    
    #Remove stop-words
    fl=[]
    for word in rem:
        if word in stop_words or not (word.isalpha()):
            continue
        else:
            fl.append(word)
    polar=findpolarity(fl, tagged) # find polarity of the words in the tweet
    # with the help of open source corpus
    
    global glen
    global hlen
    feature_extraction(fl, tagged, polar, glen) # computing the frequency of Polar Parts
    # Of Speech 
    feature_sum(fl, tagged, polar, hlen) # computing the summation of polarities 
    # Parts Of Speech wise and adding them up to find the net polarity of each tweet
    cntt+=1

#--------------------------------------------------------------------------------------

    
# keys are the count of non-POS features
keys=['handle', 'punc', 'url', 'phashtag', 'neghashtag', 'nhashtag', 'neg', '5e', '4e', '2e', '1e', 'excl']
# polar_pos are the count of Polar POS 
polar_pos=['pNN', 'nNN', '0NN', 'pJJ', 'nJJ', '0JJ', 'pRB', 'nRB', '0RB', 'pVB', 'nVB', '0VB', 'pwords', 'nwords']
# gsum is the summation of prior polarities of both Polar and Non-Polar POS.
gsum=['NN_sum', 'JJ_sum', 'VB_sum', 'RB_sum', 'total_sum']
glen = len(keys)
keys.extend(polar_pos)
hlen = len(keys)
keys.extend(gsum)
cntt=0
tweets=[]
l=[[0.0]*len(keys) for i in range(16500)]
labels=[]
with open('train/balanced.csv', 'rb') as inp:
    reader=csv.reader(inp)
    for row in reader:
        clean_doc(row[2]) # responsible for computing the sentifeatures of each tweet
        if row[1] == 'negative':
            labels.append('-1')
        elif row[1] == 'positive':
            labels.append('1')
        elif row[1] == 'neutral':
            labels.append('0')
print type(l), len(l), len(l[0])
# Saving sentifeatures
with open("train/senti.csv", "wb") as f:
    writer = csv.writer(f)    
    for row in l:
        writer.writerow(row)
# Saving labels
with open('train/labels.txt', 'wb') as op:
    json.dump(labels, op)

