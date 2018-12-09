#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Preprocessed all the tweets in "finaldataset.csv" using the module "preprocessing.py"
# and saved them in a seperate csv file "data.csv"
# preprocessing the data 
# such as converting @ to __HNDL tag, URLs processing, Hashtags processing, negations, 
# emoticons and punctuations processing


import preprocessing as p # loading the preprocessing module
import heuristics as h 
# heuristics contains all the corpuses of emoticons, sentiwords, 
# spellchecker, acronym dictionary
import csv

with open('finaldataset.csv', 'rb') as inp:
    all=[]
    reader=csv.reader(inp)
    cnt=0
    for row in reader:
        mod=[]
        mod.append(cnt)
        cnt+=1
        mod.append(row[2])
        psen = p.processAll(row[1]) # processAll is the function in "preprocessing module"
        # which converts
        for item in psen.split():
            if item.lower() in h.NEGATE:
                psen=psen.replace(item, '__NEG')
        mod.append(psen)
        all.append(mod)
print len(all)


# write the processed data to "data.csv" 
with open('data.csv', 'wb') as op:
    writer=csv.writer(op, lineterminator='\n')
    writer.writerows(all)


