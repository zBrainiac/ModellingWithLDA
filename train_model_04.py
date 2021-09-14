#######################################
#
#  20190207 Simple LDA demo build
#
#  Step 04 - visualise the LDA model
#
#######################################


ldamodel.show_topics()

#or a formatted version
pd.set_option('max_colwidth',700)
num_words=15
topic_list=ldamodel.show_topics(num_words=num_words,formatted=True)
df = pd.DataFrame(topic_list)
df

warnings.simplefilter('ignore')

import pyLDAvis

import pyLDAvis.gensim_models as gensimvis
pyLDAvis.enable_notebook()

vis_data=gensimvis.prepare(ldamodel, corpus, dictionary,sort_topics=False)

pyLDAvis.display(vis_data)

#######################################