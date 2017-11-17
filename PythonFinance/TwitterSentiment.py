# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 02:49:25 2017

@author: Akhilesh
"""

import tweepy
from textblob import TextBlob

consumer_key = 'jlKw5LWcUqhvSsNFrlllDYxSW'
consumer_secret= 'onDynGhtj1mVc9gS61oxHdvC7TmuqfSuIdEJxUgSgrM506opgN'

access_token = '931271758312751104-8T6ZOa1xfWqJYYVxvvQvYJ1MXO13YIz'
access_token_secret = '8rZfWxLMHtf1wMibkKPtKRh3PvJJLmmrchZyqJd8uuW9g'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Trump')

for tweet in public_tweets:
    print(tweet.text)
    analysis =TextBlob(tweet.text)
    print(analysis.sentiment)


'''

from textblob import TextBlob

wiki = TextBlob("I am angry that I never gets good matches on Tinder.")
print(wiki.tags)

print(wiki.words)

print(wiki.sentiment.polarity)

'''