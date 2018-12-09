from gensim.models import Word2Vec
import csv, sys, json
from numpy import array
from numpy import asarray
from numpy import zeros, argmax
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import numpy as np
import pandas
from random import shuffle
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
import tensorflow as tf
from keras.preprocessing.text import one_hot


# A word embedding is similar to a word vector

# Task 1
# construct word embeddings using word2vec and use logistic after that

# Task 2
# use your feature vectors and feed them into a neural network (a fully connected neural network)

# Task 3
# construct word embeddings using one hot encoding(or any other) and feed them into a neural network (same as above)

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

# 25485
# Task 3

labels=load_label()
with open('../binary_neutral/train/tweets.txt', 'r') as f:
    content = f.readlines()
print len(content)

content, labels = shuffleall(content, labels)

vocab_size=6894
encoded_docs = [one_hot(d, vocab_size) for d in content]


max_length = 15
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print len(padded_docs)

train_docs = padded_docs[:13200]
test_docs = padded_docs[13200:]
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=15, trainable=True))
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(units=3, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print model.summary()
# model.add(Embedding(len(words), 100, input_length=(max words in a sentence))) For Task 3
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
model.fit(np.array(train_docs), dummy_y[:13200], epochs=20, verbose=1, validation_split=0.2)

loss, accuracy = model.evaluate(np.array(test_docs), dummy_y[13200:], verbose=0)
print model.predict(np.array(test_docs))
print loss, accuracy
'''
[[0.11561069 0.4771284  0.37162   ]
 [0.14777295 0.4031027  0.3872957 ]
 [0.08985948 0.41842473 0.48500142]
 ...
 [0.39717478 0.3134149  0.31240535]
 [0.10534113 0.35304448 0.5293362 ]
 [0.1968837  0.56974417 0.21795732]]
0.9704604856450683 0.4864652937138581
'''

#1.9512323166819916 0.702105414965612 25485
#1.8451097897327307 0.707575770146919 16500
