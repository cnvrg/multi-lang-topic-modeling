# Topic Modelling (Endpoint)
Topic modeling is an unsupervised machine learning technique that’s capable of scanning a set of documents, detecting word and phrase patterns within them, and automatically clustering word groups and similar expressions that best characterize a set of documents.
## Input Arguments

In case of this library being used with topic modelling training, it receives a few inputs, which are: -
1. **'topic_df.csv'**
Descriptions and keywords which describe the topics along with their labels

    | |Topics |
    |---|---|
    |Topic 1|drive card scsi disk controller hard problem system floppy driver|
    |Topic 2|edu article writes apr article apr edu writes writes article news uiuc uiuc edu|

2. **'vectorizer.sav'**
tf-idf vectorizer which extracts features from text and has been saved after being dumped in previous library
3. **'nmf.sav'**
nmf model file from sklearn decomposition library, trained with user's data
4. **'topics_cnt_final.csv'**
file containing the number of topics the user wants to see in the output and/or the ratio by which the ideal count of topics (taken from optimizing coherence score) be modified and show in the output
    |Ratio|Number|
    |---|---|
    |0.5|0|
    or
    |Ratio|Number|
    |---|---|
    |0|3|
***Kindly mention either ratio or number as 0, because only one of them will be used.**


## Features
- input is given as text of the paragraph the user wants modelled.
- after topic modelling is run, the topics that are selected are shown as a json output

## Model Artifacts
Output
```
{"prediction":{"Dominant Topic":["window file do program use application using problem run version","god christian jesus bible christ believe faith church people say","drive scsi disk hard floppy ide controller system meg problem","car engine price bike model mile new dealer speed oil","card driver video monitor color bus bit mode graphic board","people israel gun israeli right armenian government law state jew"],"Query Doc":"I love computers windows","Topic Probability":["10.49 %","0.67 %","0.16 %","0.03 %","0.0 %","0.0 %"]}}
```
## How to run
```
import http.client

conn = http.client.HTTPSConnection("inference-1-1.azq2zmkpjtgprpbpxxklb9b.staging-cloud.cnvrg.io", 443)

import json
request_dict = {"input_params":"I love computers windows"}
payload = "{\"input_params\":" + json.dumps(request_dict) + "}"


headers = {
    'Cnvrg-Api-Key': "p1CnmrYqYzo4LF1HJpsZbumQ",
    'Content-Type': "application/json"
    }

conn.request("POST", "/api/v1/endpoints/pxyupy3pfuwdpkuaek74", payload, headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))
```
## Sample input
```
"I love computers windows. It's easy to use and cheaper compared to Apple products"
```
## References
a. LDA (Latent Diricht Allocation) (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
b. NMF (Non-Negative Matrix Factorization) (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
c. Coherence Score (https://radimrehurek.com/gensim/models/coherencemodel.html)
d. dump/load library (https://pypi.org/project/joblib/)
