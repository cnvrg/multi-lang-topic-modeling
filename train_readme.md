You can use this blueprint to train a topic model on custom data and extract key topics out of any paragraph.
In order to train this model with your data, you would need to provide one folder located in s3:
- topic_modelling: the super folder where the training file, file containing text to be decomposed (modelled) are stored.

1. Click on `Use Blueprint` button
2. You will be redirected to your blueprint flow page
3. In the flow, edit the following tasks to provide your data:

   In the `S3 Connector` task:
    * Under the `bucketname` parameter provide the bucket name of the data
    * Under the `prefix` parameter provide the main path to where the training file and model/tokenizer folders are located

   In the `train` task:
    *  Under the `input_path` parameter provide the path to the input file including the prefix you provided in the `S3 Connector`, it should look like:
       `/input/s3_connector/<prefix>/topic_modelling_data.csv`

**NOTE**: You can use prebuilt data examples paths that are already provided

4. Click on the 'Run Flow' button
5. In a few minutes you will train and deploy a topic model on your custom data.
6. Go to the 'Serving' tab in the project and look for your endpoint
8. You can also integrate your API with your code using the integration panel at the bottom of the page

Congrats! You have trained and deployed a custom model that models topics on text!