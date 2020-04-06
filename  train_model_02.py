#######################################
#
#  20190207 Simple LDA demo build
#
#  Step 02 - pre-process the data
#
#######################################

tokenizer = RegexpTokenizer(r'\w+')

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


# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:

    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    # add tokens to list
    texts.append(stemmed_tokens)

#doc_set[0]
#texts[0]

#######################################