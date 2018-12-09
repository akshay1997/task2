
SemEval-2017 Task 4: Sentiment Analysis in Twitter
http://alt.qcri.org/semeval2017/task4/

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

The data enclosed is a compilation of all annotated sentiment datasets for the five 2017 tasks. Our particular task is Task A.
It is divided by utility for a particular subtask in 2017:

A:   Message Polarity Classification
Each file includes the year and type (dev/test/train) of download to refer to the collection from prior SemEval runs of this task: in 2013-2016.
Can be found in Subtask_A/ directory
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
Summary of the task:
Given a message, classify whether the message is of "positive", "negative", or "neutral" sentiment.
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        	DATASET
        	=======

Combined the data in all the files together and extracted 16500 tweets divided into equal number of positive, negative and neutral tweets. 
Number of Positive Tweets = 5500
Number of Negative Tweets = 5500
Number of Neutral Tweets  = 5500
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	CODE PIPELINE
	==============
	
NOTE: If Balanced Dataset is already there inside "train/" directory in the csv format with name "balance.csv",
start from STEP 4
 
1) Run make_single_dataset.py
Combined the tweets in all the files in directory binary_neutral/Subtask_A/ and compiled
them into a single csv file "finaldataset.csv"

2) Run preprocess_all.py
Preprocessed all the tweets in "finaldataset.csv" using the module "preprocessing.py"
and saved them in a seperate csv file "data.csv"

3) Run balanced_dataset.py
Extracted 16500 instances (5500 belonging each class) from "data.csv" and 
saved to "train/sampletrain.csv"

4) Run extract_senti_features.py
Calculates the Senti-features tweet wise and store in "train/senti.csv"

5) Run build_vocabulary.py
Builds vocabulary on the basis of the tweets and save it in "train/vocab.txt"

6) Run vectorize.py
Firstly, preprocesses each tweet according to the vocabulary and save them in
"train/tweets.txt" and then using that preprocessed tweet converts them into 
feature vectors

7) Run train.py
Splits the training set into training and testing set ina 80:20 ratio. Cross-Validates
on the train set in order to find the optimal regularization parameter and using that
parameter trains on the train set and predicts on the test set.

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    OTHER FILES
   =============
1) heuristics.py-> All Negations, intensifiers, conjunctions as well as 
all corpuses such as acronym corpus, emoticon lookup table, 
sentiment lookup table etc. stored as dictonaries

2) preprocessing.py-> Used as a module to preprocess the tweets

3) corpus/ -> Corpus directory consists of all the Corpus used for sentiment analysis 
task
 
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

Contact:

E-mail: akshaykhanna1997@gmail.com

Akshay Khanna, Manipal Institute of Technology, Manipal

