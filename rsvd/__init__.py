"""
PyRSVD
------
This package provides a regularized singular value decomposition (RSVD) solver used to compute low-rank approximations of a large partial matrices
(i.e. a matrices with lots of missing valus).

R ~ U*V' where R: MxC, U: MxK and V: CxK. M is the number of movies, C is the number of clients

This kind of solvers have proven very successful in collaborative
filtering. In CF, such latent factor models are used to reveal the latent
structure in the dataset. The factorization produced by the solver can
directly be used to predict ratings or as a preprocessing step to e.g.
represent each user by a vector of latent topic in which he or she
is interested.

A regularized version of stochastic gradient descent is used to
minimize the approximation error measured by the squared error
of the (known) ratings and the
prediction based on the factorization.

Usage
-----


Notes
-----
It is assumed that the training data is properly shuffeled.

Requires
--------
U{Python 2.5 <http://www.python.org/download/>} or later,
U{Numpy 1.1 <http://www.numpy.org/>} or later.
"""

import sys

__version__="0.2.5"
__author__="peter.prettenhofer@gmail.com"
__license__="mit"

_numpy_version=(1,1)

try:
    import numpy as np
    if tuple((int(x) for x in np.__version__.split('.')[:2]))\
       < _numpy_version:
        print "Numpy version %s is required for PyRSVD (%d detected)." % ('.'.join(_numpy_version),np.__version__)
        sys.exit(-1)
except ImportError,e:
    print "PyRSVD requieres Numpy %s. " % '.'.join(_numpy_version)
    sys.exit(-1)

from rsvd import RSVD,rating_t
from dataset import MovieLensDataset,NetflixDataset
