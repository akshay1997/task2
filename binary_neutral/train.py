#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Splits the dataset into training and testing set ina 80:20 ratio. Cross-Validates
# on the train set in order to find the optimal regularization parameter and using that
# parameter trains on the train set and predicts on the test set.
# 80/20 Train/Test Split
# Len(Train) - 13200
# Len(Test) - 3300
import numpy as np
import csv, json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

def load_fv():
    fv=[]
    inp=open('train/allfv.csv', 'r')
    y = csv.reader(inp)
    for row in y:
        fv.append(row)
    print 'fv', len(fv)
    return fv

def load_label():
    lab=[]
    with open('train/labels.txt', 'rb') as lab:
        lab = json.load(lab)
    print 'labels', len(lab)
    return lab

# for binary (2 class classification)
def calc_pred(predicts, y_test):
    actual0pred0=0
    actual1pred0=0
    actual1pred1=0
    actual0pred1=0
    for i in range(len(predicts)):
        if predicts[i] == 0 and y_test[i] == 0:
            actual0pred0+=1
        elif predicts[i] == 0 and y_test[i] == 1:
            actual1pred0+=1
        elif predicts[i] == 1 and y_test[i] == 0:
            actual0pred1+=1
        elif predicts[i] == 1 and y_test[i] == 1:
            actual1pred1+=1
    print 'actual1pred1 = %d, actual0pred1 = %d, actual1pred0 = %d, actual0pred0 = %d\n' % (actual1pred1, actual0pred1, actual1pred0, actual0pred0)

# prints the exact probability by which the classifier guessed this
def calc_exact_pred(exact_pred, predicts, y_test_arr):
    for val in range(len(exact_pred)):
        print (exact_pred[val], predicts[val], y_test_arr[val])


# count of negative, positive, neutral tweets predicted by the clssifier
def count_pred(predicts, y_test_arr):
    cnt0=0
    cnt1=0
    cnt2=0
    for item in predicts:
        #print item, type(item)
        if item == -1.0:
            cnt0+=1
        if item == 0.0:
            cnt1+=1
        if item == 1.0:
            cnt2+=1
    print cnt0, cnt1, cnt2
    cnt0=0
    cnt1=0
    cnt2=0
    for item in y_test_arr:
        if item == -1.0:
            cnt0+=1
        if item == 0.0:
            cnt1+=1
        if item == 1.0:
            cnt2+=1
    #print cnt0, cnt1, cnt2
            

fvec=load_fv()
labels=load_label()
# to make the train-test similar to the paper ratio - 0.69, 0.84
#train_test split
fvec_train, fvec_test, lab_train, lab_test = train_test_split(fvec, labels, test_size=0.20, random_state=1)
fvec_test, fvec_valid, lab_test, lab_valid = train_test_split(fvec_test, lab_test, test_size=0.0, random_state=1)
#convert to np arrays
X_train_arr=np.array(fvec_train).astype(float)
X_test_arr=np.array(fvec_test).astype(float)
#X_valid_arr=np.array(fvec_valid).astype(float)
y_train_arr = np.array(lab_train).astype(float)
y_test_arr = np.array(lab_test).astype(float)
#y_valid_arr = np.array(lab_valid).astype(float)
print len(X_train_arr), len(X_test_arr)
regularization_param_grid = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 1, 1.5, 2, 10]
acc=0.0
# 5 fold cross-validation on the training set to find the best class parameter 
# C
for iter in regularization_param_grid:
    clf=LogisticRegression(C=iter)
    score=(cross_val_score(clf, X_train_arr, lab_train, cv=5))
    print score
    score=score.mean()
    print score, iter
    if score > acc:
        acc=score
        maxiter=iter


clf=LogisticRegression(C=maxiter)
clf.fit(X_train_arr, y_train_arr)
predicts = clf.predict(X_test_arr)
exact_pred = clf.predict_proba(X_test_arr)
#calc_exact_pred(exact_pred, predicts, y_test_arr)
#calc_pred(predicts, y_test_arr)
count_pred(predicts, y_test_arr)
print 'test accuracy'
print accuracy_score(predicts, y_test_arr), maxiter
