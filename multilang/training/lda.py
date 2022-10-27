import pandas as pd
import re
import gensim
from gensim.utils import simple_preprocess
import spacy
import gensim.corpora as corpora
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.models import CoherenceModel
import numpy as np
import tqdm
import argparse
import os
from random import randrange
import joblib

parser = argparse.ArgumentParser(description="""Creator""")
parser.add_argument(
    "-f",
    "--training_file",
    action="store",
    dest="training_file",
    default="/data/topic_data/sample_topic_data.csv",
    required=True,
    help="""location of input file""",
)
parser.add_argument(
    "--topic_size",
    action="store",
    dest="topic_size",
    default="8,15,2",
    required=True,
    help="""min, max, stepsize / number_of_topics""",
)
parser.add_argument(
    "--chunk_size",
    action="store",
    dest="chunk_size",
    default="100",
    required=True,
    help="""Number of documents to be used in each training chunk.""",
)
parser.add_argument(
    "--passes",
    action="store",
    dest="passes",
    default="1",
    required=True,
    help="""Number of passes through the corpus during training.""",
)
parser.add_argument(
    "--eta",
    action="store",
    dest="eta",
    default="symmetric",
    required=True,
    help="""A-priori belief on topic-word distribution""",
)
parser.add_argument(
    "--alpha",
    action="store",
    dest="alpha",
    default="symmetric",
    required=True,
    help="""A-priori belief on document-topic distribution""",
)

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")
args = parser.parse_args()
input_data = pd.read_csv(args.training_file)
topic_size = args.topic_size.split(",")
chunk_var=int(args.chunk_size)
pass_var=int(args.passes)
eta_var=args.eta
alpha_var=args.alpha

# Breaking up the wikipedia output text into sentence level
textual_df = input_data[['text']]
sample = ['4']
concatenated_df = pd.DataFrame()
temp_df = pd.DataFrame(sample, columns=['text'])
for i in range(textual_df.shape[0]):
    subset = textual_df['text'][i].split(' ')
    print(subset)
    print('checkdsf')
    p=0
    q=0
    count = int(len(subset)/10)
    for j in range(count):
        try:
            print(q)
            print('check8')
            p = q+randrange(13, 16)
            temp_df.loc[j] = " ".join(subset[q:p])
            print(temp_df)
            q = p
        except:
            print('error')
    temp_df = temp_df.replace(r'^s*$', float('NaN'), regex=True)
    temp_df.dropna(inplace=True)
    concatenated_df = pd.concat([concatenated_df, temp_df])
df = concatenated_df.reset_index()
df.to_csv('/cnvrg/df.csv')
# Text Cleaning
df['text']=df['text'].apply(str)
# Remove punctuation
df['text_processed'] = df['text'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the titles to lowercase
df['text_processed'] = df['text_processed'].map(lambda x: x.lower())

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data = df.text_processed.values.tolist()
data_words = list(sent_to_words(data))

#Phrase Modeling: Bigram and Trigram Models
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Topics range
min_topics = int(float(topic_size[0]))
max_topics = int(float(topic_size[1]))
step_size = int(float(topic_size[2]))
topics_range = range(min_topics, max_topics, step_size)

random_state_var = randrange(100)

def compute_coherence_values(corpus, dictionary, k):
    
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=random_state_var,
                                           chunksize=chunk_var,
                                           passes=pass_var,
                                           eta=eta_var,
                                           alpha=alpha_var)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    return coherence_model_lda.get_coherence()

model_results = pd.DataFrame(columns=['Topics_Count','Coherence'])
model_results.to_csv('/cnvrg/model_results_prem.csv')
iterator = 0
for k in topics_range:
    cv = compute_coherence_values(corpus=corpus, dictionary=id2word,k=k)
    # Save the model results
    model_results.at[iterator,'Topics_Count'] = k
    model_results.at[iterator,'Coherence'] = cv
    iterator= iterator+1

model_results = model_results.sort_values(by='Coherence',ascending=False)
model_results = model_results.reset_index()
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=model_results['Topics_Count'][0],
                                       random_state=random_state_var,
                                       chunksize=chunk_var,
                                       passes=pass_var,
                                       eta=eta_var,
                                       alpha=alpha_var)

model_file = cnvrg_workdir+'/lda_model.sav'
corp_dict_file = cnvrg_workdir+'/corp_dict.sav'
model_results_file = cnvrg_workdir+'/model_results.csv'
model_results.to_csv(model_results_file)
joblib.dump(lda_model, model_file)
joblib.dump(id2word, corp_dict_file)
#final_df = pd.DataFrame(model_results).sort_values(by="Coherence", ascending=False)
#final_df_file = cnvrg_workdir + '/topics_cnt_file.csv'
#final_df.to_csv(final_df_file)
