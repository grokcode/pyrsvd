.. _api:

==========
PyRSVD API
==========

The :mod:`rsvd` Module: A regularized SVD solver for collaborative filtering
----------------------------------------------------------------------------

.. module:: rsvd
   :platform: Unix, Windows
   :synopsis: Regularized SVD solver for collaborative filtering

.. moduleauthor:: Peter Prettenhofer <peter.prettenhofer@gmail.com>

.. attribute:: rating_t
    
   The data type of the rating arrays. It is a (uint16,uint32,float) triple. A :class:`numpy.dtype` object. 

.. class:: RSVD
   
   .. function:: train(factors,ratingsArray,dims,probeArray=None, maxEpochs=100,minImprovement=0.000001, learnRate=0.001, regularization=0.011, randomize=False)

      `factors` 

          The number of latent factors :math:`k` of the model. 
   
      `ratingsArray` 

          An array containing the training rating triples represented as a :class:`numpy.ndarray` of data type :const:`rsvd.ratings_t`.

      `dims`: A (num_movies,num_users) tuple.

      `probeArray`

          An array containing the probe rating triples represented as a :class:`numpy.ndarray` of data type :const:`rsvd.ratings_t`. If the argument is not `None` early stopping is performed (i.e. in each iteration the error on the probe data is computed and the training procedure is stopped as soon as the difference between the current and the last probe error is smaller than `minImprovement`. 

      `maxEpochs`

          The maximum number training sweeps to perform (i.e. the maximum number of iterations of the outer loop of the algorihtm). 

      `minImprovement`

          The minimum improvement in probe error in order to continue training. (see argument `probeArray`). Is used iff `probeArray!=None`. 

      `learnRate` 

          The learning rate parameter of the gradient descent procedure. 

      `regularization`

          The regularization parameter :math:`\lambda` of the regularized error function. The higher the regularization term the more are large parameters penalized. 

      `randomize`

          Whether or not the training data should be shuffeled each 10th iteration (via :meth:`numpy.random.shuffle`). Note: If the ratings array is huge, this may take a while. 

   .. method:: RSVD.save(model_dir)
