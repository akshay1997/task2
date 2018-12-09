# Combined the tweets in all the files in diectory binary_neutral/Subtask_A/ and compiled
# them into a single csv file "finaldataset.csv"
import os, csv

def make_Corpus(root_dir):
    polarity_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]        
    with open('finaldataset.csv', 'wb') as filew:
        writer=csv.writer(filew)
        all=[]
        cnt=0
        for polarity_dir in polarity_dirs:
            item=polarity_dir.split('.')
            #print item
            if item[1] == 'txt':
                with open(polarity_dir, 'rb') as file:
                    lines = file.readlines()
                    for line in lines:
                        row=[]
                        line=line.rstrip('\n').split('\t')
                        #print line
                        row.append(cnt)
                        #print line[2], polarity_dir
                        row.append(line[2])
                        row.append(line[1])
                        all.append(row)
                        cnt+=1
            elif item[1] == 'tsv':
                row=[]
                with open(polarity_dir, 'rb') as file:
                    reader=csv.reader(file)
                    for line in reader:
                        s=' '.join(line).rstrip('\n').split('\t')
                        #print s
                        row=[]
                        row.append(cnt)
                        row.append(s[3])
                        row.append(s[2])
                        all.append(row)
                        cnt+=1
        writer.writerows(all)
    print len(all)
        


root_dir = 'twitter/binary_neutral/Subtask_A'
corpus = make_Corpus(root_dir)
