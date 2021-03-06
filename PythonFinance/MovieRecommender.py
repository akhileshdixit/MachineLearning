# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 08:26:25 2017

@author: Akhilesh
"""

# Two types - Collaborative (what other liked in past) and Content based (what you liked in past).
# Hybrid = Collaborative + Content 

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# Fetch data and format it
data = fetch_movielens(min_rating=4.0)

#for dkey, dval in data.items():
#    print(dkey, dval)

# print training and testing data
print (repr(data['train']))
print (repr(data['test']))

# Create model - warp: Weighted Approximate-Rank Pairwise
# Use Gradient Descedent Algorithm
model = LightFM(loss='warp')

# Train model
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, user_ids):

    #number of users and movies in training data
    n_users, n_items = data['train'].shape

    #generate recommendations for each user we input
    for user_id in user_ids:

        #movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        #rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        #print out the results
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)

sample_recommendation(model, data, [3, 25, 450])
