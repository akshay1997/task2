import numpy as np
import csv, json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_fv():
    fv=[]
    inp=open('../binary_neutral/train/allfv.csv', 'r')
    y = csv.reader(inp)
    for row in y:
        fv.append(row)
    print 'fv', len(fv)
    return fv

def load_label():
    lab=[]
    with open('../binary_neutral/train/labels.txt', 'rb') as lab:
        lab = json.load(lab)
    print 'labels', len(lab)
    return lab
#----------------------------------------------------------------------------------------#

fvec=load_fv()
labels=load_label()
fvec_train, fvec_test, lab_train, lab_test = train_test_split(fvec, labels, test_size=0.20, random_state=1)
fvec_test, fvec_valid, lab_test, lab_valid = train_test_split(fvec_test, lab_test, test_size=0.0, random_state=1)
vocab_size =  len(fvec_train[0]) # vocab size = 5209



X_train_arr=np.array(fvec_train).astype(float)
X_test_arr=np.array(fvec_test).astype(float)
#X_valid_arr=np.array(fvec_valid).astype(float)
#y_train_arr = np.array(lab_train).astype(float)
#y_test_arr = np.array(lab_test).astype(float)
#y_valid_arr = np.array(lab_valid).astype(float)
print len(X_train_arr), len(X_test_arr)

encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

model=Sequential()
model.add(Dense(units=256))
model.add(Dense(128))
model.add(Dense(32))

model.add(Dense(8))
model.add(Dense(3, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(X_train_arr, dummy_y[:13200], epochs=30, verbose=1, validation_split=0.1)
print(model.summary())
'''
# evaluate the model
loss, accuracy = model.evaluate(np.array(padded_docs[:13200]), dummy_y[:13200], verbose=0)
print('Train Accuracy: %f' % (accuracy*100)) #train accuaracy
'''

loss, accuracy = model.evaluate(X_test_arr, dummy_y[13200:], verbose=0)
print('Test Accuracy: %f' % (accuracy*100)) #Accuracy: 53.001178


