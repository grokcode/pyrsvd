.. _overview:

===============
PyRSVD Overview
===============

Introduction
============

PyRSVD provides an efficient python implementation of a regularized 
singular value decomposition solver. The module is primarily aimed at 
applications in collaborative filtering, in particular the `Netflix 
competition <http://netflixprize.com>`_. 

Matrix Approximation
====================

The solver is used to compute a low-rank approximation of a rating matrix ''R'', which is usually a large partial matrix (i.e. lots of missing values).

More formally,

.. math::

    \mathbf{R} \approx \mathbf{U} \mathbf{V}^T \text{ where, } \mathbf{R}: m \times n, \mathbf{U}: m\times k, \mathbf{V}:n\times k

The goodness of the approximation is measured in terms of the frobenius norm :math:`\lVert \mathbf{R}-\mathbf{U} \mathbf{V}^T\rVert_{F}` with respect to the known ratings. Minimizing the frobenius norm between the rating matrix R and the factorization is equivalent to minimize the squared error. Due to the huge number of parameters, overfitting is a serious problem. It is avoided by adding a regularization term to the squared error function, which penalizes large parameters. 
The regularized error function is given by, 

.. math::
   
    \mathcal{L}(\mathbf{U},\mathbf{V}) = \sum_{i,j \in R} (r_{i,j} - \mathbf{U}_i \  \mathbf{V}_j^T) + \lambda (\sum_i \lVert \mathbf{U}_i \rVert^2 + \sum_j \lVert \mathbf{V}_j \rVert^2)

The solver uses stochastic gradient decent to minimize the above error function. 

Matrix approximation is a simple proven technique for collaborative filtering. The factors reveal some of the latent structure in the rating data which is subsequently used to predict user preferences. The factorization produced by the solver can
directly be used to predict ratings or as a preprocessing step, e.g. to represent each user by a vector of latent factors he or she is interested in.




Future Plans
============
 * Extend the interface to allow both, uint8 and float ratings. 


