#######################################
#
#  20190207 Simple LDA demo build
#
#  Step 03 - build the LDA model
#
#######################################


# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
# Set training parameters.
num_topics = 5
chunksize = 2000
passes = 20
iterations = 1000

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics, id2word = dictionary, passes=passes, random_state=10)

#save the trained model
ldamodel.save("models/lda_model_trained")
dictionary.save("models/lda_model_dictionary")

#######################################