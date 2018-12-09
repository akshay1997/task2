# Firstly, preprocesses each tweet according to the vocabulary and save them in
# "train/tweets.txt" and then using that preprocessed tweet converts them into 
# feature vectors
import json, csv
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def clean_doc(doc):
    # select words which are english words, not stop words and of length 
    # atleast 1
    tokens = doc.split()
    #doc=doc.lower()
    rem=[]
    for word in tokens:
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
            pass
        else:
            rem.append(word)
    tokens=rem
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def tweet_to_line(doc, vocab):
    # converts the preprocessed tweet into a form which can be vectorised
    tokens = clean_doc(doc)
    tokens = [w for w in tokens if w in vocab] # consider those words which are 
    # present in vocab
    return ' '.join(tokens)

def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()
    
def vectorize(st, corpus, vocab):
    # Using the processed tweets, construct feature vectors for those tweets
    vocab = list(set(vocab))
    print len(vocab)
    corpus = dict(((term, index) for index, term in enumerate(sorted(vocab))))
    num_words = len(corpus)
    senti=[]
    with open("%s/senti.csv" % st, "rb") as f:
        reader = csv.reader(f)    
        for row in reader:
            senti.append(row)
        #print 'senti', len(senti)
    with open('%s/tweets.txt' % st, 'r') as f:
        content = f.readlines()
        fvs = [[0]*num_words for x in range(len(content))]
    with open("%s/allfv.csv" % st, "wb") as f2:
        writer = csv.writer(f2)    
        for x in range(len(content)):
            fv = fvs[x]
            # if a word w found in the processed tweet, feature_vector[word] 
            # increments by 1 -> frequency based feature values
            for word in content[x].split():
                fv[corpus[word]]+=1 
            fv.extend(senti[x])
            writer.writerow(fv)
            #print len(fv)
        

    
    
with open('train/vocab.txt', 'rb') as op:
   vocab=json.load(op)

trainlines=[]
testlines=[]
with open('train/balanced.csv', 'rb') as inp:
    reader=csv.reader(inp)
    for row in reader:
        trainlines.append(tweet_to_line(row[2], vocab))
save_list(trainlines, 'train/tweets.txt') # processed tweets writing to file

vectorize("train",trainlines, vocab) # using the processed tweets to vectorise
# them 