#######################################
#
#  20190207 Simple LDA demo build
#
#  Step 05 - predict topics
#
#######################################

import warnings
warnings.simplefilter('ignore')

import pandas as pd

import numpy as np

import gensim

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models

# create English stop words list
en_stop = get_stop_words('en')

#add some custom stop_words
en_stop.append(u'xxxx')
en_stop.append(u'xx')
en_stop.append(u'sp')
en_stop.append(u'00')
en_stop.append(u't')
en_stop.append(u'n')
en_stop.append(u'c')
en_stop.append(u's')


tokenizer = RegexpTokenizer(r'\w+')
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# Load the trained model
ldamodel_t=gensim.models.LdaModel.load("models/lda_model_trained")
dictionary_t=gensim.corpora.Dictionary.load("models/lda_model_dictionary")

def clean_text(text):
    # clean and tokenize document string
    lower_text = text.lower()
    tokens = tokenizer.tokenize(text)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    return stemmed_tokens

def score_text(text):
    clean=clean_text(text)
    bow = dictionary_t.doc2bow(clean)

    text_topics=ldamodel_t[bow]

    most_likely_topic = max(text_topics,key=lambda x:x[1])
    most_likely_topic_index=text_topics.index(most_likely_topic)

    #return {"clean_text":clean,"most_likely_topic":most_likely_topic}

    return {"clean_text":clean,"topics":text_topics}

def name_topics(text):

  # the topic list in the lda model in gensim is zero based and pyldavis is one based
  # this function makes them consistent and also allows you to supply topic names

  named_topics=text.replace("0,","'1: Loans',")
  named_topics=named_topics.replace("1,","'2: Debt',")
  named_topics=named_topics.replace("2,","'3: Credit Report',")
  named_topics=named_topics.replace("3,","'4: General',")
  named_topics=named_topics.replace("4,","'5: Payments',")

  return named_topics

def predict(args):
  phrase = args["phrase"]
  topic = score_text(phrase)
  topics_s = str(topic["topics"])
  topics = name_topics(topics_s)

  return {"topics": topics}

#Samples
#predict({"phrase":"my credit report is completely inaccurate"})
#predict({"phrase":"it is not accurate and i want it removed"})
#predict({"phrase": "it was delayed and now i have been charged"})
#predict({"phrase":"when i saw the statement i was shocked and immediately phoned"})
#predict({"phrase": "why do you keep refusing my application"})

#######################################