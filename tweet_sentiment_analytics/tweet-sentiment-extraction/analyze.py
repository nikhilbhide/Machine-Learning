import boto3
import json
import pandas as pd

def get_sentiment(text):
    response = json.dumps(comprehend.detect_sentiment(Text=text, LanguageCode='en'), sort_keys=True, indent=4)
    sentiment = json.loads(response)
    return sentiment


df = pd.read_csv("test.csv")
comprehend = boto3.client(service_name='comprehend', region_name='us-east-1')
for index, row in df.iterrows():
    text = row.text
    id = row.textID
    actualSentiment = row.sentiment
    predictedSentiment = get_sentiment(text)["Sentiment"].lower()
    if(actualSentiment!=predictedSentiment):
        print("discreprancy found")
        print("actual sentiment of {0} is {1} and predicted sentiment is {2}".format(text,actualSentiment,predictedSentiment))
    