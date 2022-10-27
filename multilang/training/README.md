# Latent Dirichlet allocation (Train)
Latent Dirichlet Allocation (LDA) is a generative statistical model that explains a set of observations through unobserved groups, and each group explains why some parts of the data are similar. The LDA is an example of a topic model. Observations (e.g., words) are collected into documents, and each word's presence is attributable to one of the document's topics. Each document will contain a small number of topics.

Purpose of this library is to not only do topic modelling but also calculate coherence scores based on the number of topics that will be outputted for each input data point. Optimization is done on the basis of optimum coherence score and eventually, that optimum number of topics is sent to the next library in the blueprint.

### Features
- upload the file in csv format with each row being a single document and fill the number of topics according to the user's need.
- User will get custom-trained weights for each document related to each topics.
- the library removes stopwords, bigrammatiztion, trigrammatization and lemmatization before applying the LDA algorithm

# Input Arguments
- `--input_file` refers to the name of the path of the directory where files which need to be classified are stored.
    |Documents|Label|
    |The real question here in my opinion is what Motorola processors running system 7 on a MAC are comparable to what Intel processors running Windows on a PC?  I recall there being a conversation here that a 486/25 running Windows benchmarks at about the same speed as 25Mhz 030 in system 7.  I don't know if that is true, but I would love to hear if anyone has any technical data on this. David| 4|
    |Please could someone in the US give me the current street  prices on the following, with and without any relevant taxes:   8 Mb 72 pin SIMM 16 Mb 72 pin SIMM (both for Mac LC III)  Are any tax refunds possible if they are to be exported to the UK? Can you recommend a reliable supplier?| 4|

- `--topic_size` refers to a string of 3 comma separated values of topic sizes, on which to test the lda algorithm. minumum, maximum and the step size which will increment the minimum value for testing.
8,15,2 means the values tested will be "8,10,12,14"
Recommended to use bigger step sizes

# Model Artifacts
- `--topics_cnt_file.csv` the file that contains the mapping of the each topic count (6,7,10 etc) with a coherence score. It's sorted so the top count of topics will be the one having the highest coherence score.
    |Topics_Count|Coherence|
    |---|---|
    |8|0.52|
    |14|0.51|
    |12|0.50|

# References
a. LDA (Latent Diricht Allocation) (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
b. NMF (Non-Negative Matrix Factorization) (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
c. Coherence Score (https://radimrehurek.com/gensim/models/coherencemodel.html)
d. dump/load library (https://pypi.org/project/joblib/)
