from gensim.models import Word2Vec
import csv, sys, json
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
import numpy as np
import pandas
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# A word embedding is similar to a word vector

# Task 1
# construct word embeddings using word2vec and use logistic after that

# Task 2 not possible
# use your feature vectors and feed them into a neural network (a fully connected neural network)

# Task 3  accuracy around 42
# construct word embeddings using one hot encoding(or any other) and feed them into a neural network (same as above)
def load_label():
    labels=[]
    with open('balanced2.csv', 'rb') as inp:
        reader=csv.reader(inp)
        for row in reader:
            clean_doc(row[2]) # responsible for computing the sentifeatures of each tweet
            if row[1] == 'negative':
                labels.append('-1')
            elif row[1] == 'positive':
                labels.append('1')
            elif row[1] == 'neutral':
                labels.append('0')
    return labels

# Task1 
 # refer https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

with open('tweets2.txt', 'r') as f:
    content = f.readlines()
print len(content)
docs=[]
for item in content:
    docs.append(item.split())
print len(docs), docs
model = Word2Vec(docs, min_count=1, size=100)
print model
words = list(model.wv.vocab)
print len(words)
print len(model['sentence']), model['sentence']
model.save('small.bin')
# Taken all 53484 sentences into embedding
#model = Word2Vec(sentences, min_count=1) #use window=10
#print(model)
#words = list(model.wv.vocab)

#print(len(model['sentence']))

# save models
#model.save('model.bin')
# load model

#new_model = Word2Vec.load('model.bin')
#print(new_model)

# Task 3
'''
labels=load_label()
with open('../binary_neutral/train/tweets.txt', 'r') as f:
    content = f.readlines()
print len(content)

vocab_size=10000
encoded_docs = [one_hot(d, vocab_size) for d in content]
print len(encoded_docs)

max_length = 10
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print len(padded_docs)

train_docs = padded_docs[:13200]
test_docs = padded_docs[13200:]
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=10))
model.add(Flatten())
model.add(Dense(units=16))
model.add(Dense(units=3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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