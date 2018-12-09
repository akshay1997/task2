import numpy as np
import sys
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras import regularizers
from random import shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import csv, json, random
from keras.preprocessing.text import one_hot
from keras.utils import np_utils
# define documents
with open('../binary_neutral/train/tweets.txt', 'r') as f:
    docs = f.readlines()


def shuffleall(docs, labels):
    print type(labels[0])
    part=[]
    for i in range(len(docs)):
        part.append(" "+(labels[i]))

    for i in range(len(docs)):
        docs[i]=docs[i].rstrip()
        docs[i]+=part[i]
    shuffle(docs)
    lab=[]
    for i in range(len(docs)):
        if(docs[i][-2]=='-'):
            lab.append('-1')
        elif docs[i][-1] == '0':
            lab.append('0')
        else:
            lab.append('1')
    for i in range(len(docs)):
        if(docs[i][-2]=='-'):
            docs[i]=docs[i][:-3].strip()
        else:
            docs[i]=docs[i][:-2].strip()
    
    for i in range(len(docs)):
        docs[i] += '\n'
    return docs, lab




def load_label():
    labels=[]
    with open('../binary_neutral/train/balanced.csv', 'rb') as inp:
        reader=csv.reader(inp)
        for row in reader:
            if row[1] == 'negative':
                labels.append('-1')
            elif row[1] == 'positive':
                labels.append('1')
            elif row[1] == 'neutral':
                labels.append('0')
    return labels
# define class labels
labels=load_label()
# prepare tokenizer
#print docs[:10], labels[:10]
docs, labels = shuffleall(docs, labels)
#print 'new'
#print docs[:10], labels[:10]
#print labels
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
print vocab_size
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
#print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 10
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
#print(padded_docs)
# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
# define model
model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=10, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(3, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model

encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

model.fit(np.array(padded_docs[:13200]), dummy_y[:13200], epochs=30, verbose=1, validation_split=0.1)
'''
# evaluate the model
loss, accuracy = model.evaluate(np.array(padded_docs[:13200]), dummy_y[:13200], verbose=0)
print('Train Accuracy: %f' % (accuracy*100)) #train accuaracy
'''

loss, accuracy = model.evaluate(np.array(padded_docs[13200:]), dummy_y[13200:], verbose=0)
print('Test Accuracy: %f' % (accuracy*100)) #Accuracy: 53.001178

