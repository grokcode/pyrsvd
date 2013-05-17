.. _tutorial:

========
Tutorial
========

Install PyRSVD
==============

Install PyRSVD::

   sudo python setup.py install

Load the MovieLens dataset
==========================

Download the MovieLens dataset from [GroupLens]_ and extract the archive to `/path/to/data`. 
Load the dataset::
   
   import numpy as np
   from rsvd import MovieLensDataset
   dataset = MovieLensDataset.loadDat('data/ratings.dat')
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
   
Train a model
=============

The following code trains a model with 20 factors, learn rate 0.0005 and regularization term 0.005 ::

   from rsvd import RSVD
   dims = (dataset.movieIDs().shape[0], dataset.userIDs().shape[0])
   model = RSVD.train(20, train, dims, probeArray=val, learnRate=0.0005, regularization=0.005)

Evaluate the model
==================

Now we will evaluate the trained model on the test set::

   sqerr=0.0
   for movieID,userID,rating in test:
       err = rating - model(movieID,userID)
       sqerr += err * err
   sqerr /= test.shape[0]
   print "Test RMSE: ", np.sqrt(sqerr)


Save the model
==============

You can save the model to disk simply by calling the RSVD.save method. ::

    model.save('path/to/model/dir')

.. [GroupLens] GroupLens 1M ratings datasets, http://grouplens.org/system/files/million-ml-data.tar__0.gz