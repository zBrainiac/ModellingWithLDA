#######################################
#
#  20190207 Simple LDA demo build
#
#  Step 01 - read the data
#
#######################################

import warnings
warnings.simplefilter('ignore')

import pandas as pd

import numpy as np

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models

import gensim
pd.set_option('max_colwidth',40)

# load file, select number of rows for training
training_file="consumer_complaints_16952.csv"
training_rows=500

df = pd.read_csv(training_file,nrows=training_rows) #Reading the dataset in a dataframe using Pandas
#df.head(5)

doc_set=df["Consumer complaint narrative"]
#doc_set[0]

#######################################