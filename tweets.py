import tweepy
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import pathlib
import keys 

# Consumer keys and access tokens, used for OAuth
consumer_key = keys.ck
consumer_secret = keys.cs
access_token = keys.at
access_token_secret = keys.ats

# OAuth process, using the keys and tokens
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
 
# Create the actual interface using authentication
api = tweepy.API(auth)
sid = SentimentIntensityAnalyzer()

def get_tweets(search_terms):
	cwd = pathlib.Path.cwd()
	path = cwd/"tweets"
	if not path.exists():
		path.mkdir()
	for term in search_terms:
		tweets = api.search(term, count=5)
		df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
		
		
		df['Sentiment'] = df['Tweets'].apply(lambda x: sid.polarity_scores(x))
		df['Good_Bad'] = df['Sentiment'].apply(lambda x: x['compound'])
		
		df.to_csv(path.joinpath(f"{term}.csv"))
		print(df)
