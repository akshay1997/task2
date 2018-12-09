# Make the imbalanced dataset in "data.csv" balanced by extracting equal number
# of positive, negative and neutral tweets and save inside train/ directory as
# "balanced.csv"

import csv

with open('data.csv', 'rb') as inp:
    op1 = open('balanced2.csv', 'wb')
    writer1=csv.writer(op1, delimiter=',', lineterminator='\n')
    reader=csv.reader(inp)
    alltrain=[]
    pos=0
    neg=0
    neu=0
    cnt=0
    for row in reader:
        if row[1] == 'negative' and neg < 8495:
            cnt+=1
            neg+=1
            t=[]
            
            t.append(cnt)
            t.append(row[1])
            t.append(row[2])
            alltrain.append(t)
            
            
        elif row[1] == 'positive' and pos < 8495:
             cnt+=1
             pos+=1
             t=[]
             t.append(cnt)
             t.append(row[1])
             t.append(row[2])
             alltrain.append(t)
            
        elif row[1] == 'neutral' and neu < 8495:
            cnt+=1
            neu+=1
            t=[]
            t.append(cnt)
            t.append(row[1])
            t.append(row[2])
            alltrain.append(t)
           
            
        if neg == 8495 and pos == 8495 and neu == 8495:
        #number of positive tweets = negative tweets = neutral tweets = 5500
            break
    writer1.writerows(alltrain)

print len(alltrain)        
