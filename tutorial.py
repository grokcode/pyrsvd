#!/usr/bin/python
"""MovieLense Tutorial

see doc/tutorial.rst for more information.
"""

import numpy as np
from rsvd import MovieLensDataset
dataset = MovieLensDataset.loadDat('data/movielense/ratings.dat')
ratings=dataset.ratings()

# make sure that the ratings a properly shuffled
np.random.shuffle(ratings)

# create train, validation and test sets.
n = int(ratings.shape[0]*0.8)
train = ratings[:n]
test = ratings[n:]
v = int(train.shape[0]*0.9)
val = train[v:]
train = train[:v]

from rsvd import RSVD
dims = (dataset.movieIDs().shape[0], dataset.userIDs().shape[0])
model = RSVD.train(20, train, dims, probeArray=val,
                   learnRate=0.0005, regularization=0.005)

sqerr=0.0
for movieID,userID,rating in test:
    err = rating - model(movieID,userID)
    sqerr += err * err
sqerr /= test.shape[0]
print "Test RMSE: ", np.sqrt(sqerr)
