# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# filename: rsvd.pyx

cimport numpy as np

import numpy as np
import sys
import pickle
import getopt
import simseq


from time import time
from os.path import exists

# Author is differrent.
__authors__ = [
      '"Peter Prettenhofer" <peter.prettenhofer@gmail.com>'
]

"""The numpy data type of a rating array, a (movieID,userID,rating) triple.
"""
rating_t = np.dtype("H,I,f4")

class RSVD(object):
    """A regularized singular value decomposition solver.
    
    The solver is used to compute the low-rank approximation of large partial
    matrices.

    To train a model use the following factory method:
    > model=RSVD.train(10,ratings,(17770,480189))

    Where ratings is a numpy record array of data type `rsvd.rating_t`, which
    corresponds to (uint16,uint32,float32).
    It is assumed that item and user ids are properly mapped to the interval [0,max item id] and [0,max user id], respectively.
    Min and max ratings are estimated from the training data. 

    To predict the rating of user i and movie j use:
    > model(j,i)
    
    """

    def __init__(self):
        pass

    def __getstate__(self):
        return {'num_users':self.num_users,
                'num_movies':self.num_movies,
                'factors':self.factors,
                'lr':self.lr,
                'reg':self.reg,
                'min_improvement':self.min_improvement,
                'max_epochs':self.max_epochs,
                'min_rating':self.min_rating,
                'max_rating':self.max_rating}

        
    def save(self,model_dir_path):
        """Saves the model to the given directory.
        The method raises a ValueError if the directory does not exist
        or if there is already a model in the directory.

        Parameters
        ----------
        model_dir_path : str
            The directory of the serialized model.
        
        """
        if exists(model_dir_path+'/v.arr') or \
           exists(model_dir_path+'/u.arr') or \
           exists(model_dir_path+'/model'):
            raise ValueError("There exists already a"+\
                             "model in %s" % model_dir_path) 

        if not exists(model_dir_path):
            raise ValueError("Directory %s does not exist." % model_dir_path)

        try:
            self.u.tofile(model_dir_path+"/u.arr")
            self.v.tofile(model_dir_path+"/v.arr")
            f=open(model_dir_path+"/model",'w+')
            pickle.dump(self,f)
            f.close()
        except AttributeError, e:
            print "Save Error: Model has not been trained.",e
        except IOError, e:
            print "IO Error: ",e

    @classmethod
    def load(cls,model_dir_path):
        """Loads the model from the given directory.

        Parameters
        ----------
        model_dir_path : str
            The directory that contains the model.

        Returns
        -------
        describe : RSVD
            The deserialized model. 
        """
        f=file(model_dir_path+"/model")
        model=pickle.load(f)
        f.close()
        model.v=np.fromfile(model_dir_path+"/v.arr").\
                 reshape((model.num_users,model.factors))
        model.u=np.fromfile(model_dir_path+"/u.arr").\
                 reshape((model.num_movies,model.factors))
        return model

    def __call__(self,movie_id,user_id):
        """Predict the rating of user i and movie j.
        The prediction is the dot product of the user
        and movie factors, resp.
        The result is clipped in the range [1.0,5.0].
        
        Parameters
        ----------
        movie_id : int
            The raw movie id of the movie to be predicted.
        user_id : int
            The mapped user id of the user. 
            The mapping is based on the sorted order of user ids
            in the training set.

        Returns
        -------
        describe : float
            The predicted rating.
            
        """
        min_rating=self.min_rating
        max_rating=self.max_rating
        r=np.dot(self.u[movie_id-1],self.v[user_id])
        if r>max_rating:
            r=max_rating
        if r<min_rating:
            r=min_rating
        return r

    @classmethod
    def train(cls,factors,ratingsArray,dims,simtx, probeArray=None,\
                  maxEpochs=100,minImprovement=0.000001,\
                  learnRate=0.001,regularization=0.011,\
                  randomize=False, randomNoise=0.005, nmfflag=True,\
                  gamma=0., randseed=None):
        """Factorizes the given partial rating matrix.

        train(factors,ratingsArray,dims,**kargs) -> RSVD

        If a validation set (probeArray) is given, early stopping is performed
        and training stops as soon as the relative improvement on the validation
        set is smaller than `minImprovement`.
        If `probeArray` is None, `maxEpochs` are performed.

	The complexity of the algorithm is O(n*k*m), where n is the number of
	non-missing values in R (i.e. the size of the `ratingArray`), k is the
	number of factors and m is the number of epochs to be performed. 

        Parameters
        ----------
        factors: int
            The number of latent variables. 
        ratingsArray : ndarray
            A numpy record array containing the ratings.E
            Each rating is a triple (uint16,uint32,float32). 
        dims : tuple
            A tuple (numMovies,numUsers).
            It is used to determine the size of the
            matrix factors U and V.
        probeArray : ndarray
            A numpy record array containing the ratings
            of the validation set. (None)
        maxEpochs : int
            The maximum number of gradient descent iterations
            to perform. (100)
        minImprovement : float
            The minimum improvement in validation set error.
            This triggers early stopping. (0.000001)
        learnRate : float
            The step size in parameter space.
            Set with caution: if the lr is too high it might
            pass over (local) minima in the error function;
            if the `lr` is too low the algorithm hardly progresses. (0.001) 
        regularization : float
            The regularization term.
            It penalizes the magnitude of the parameters. (0.011)
        randomize : {True,False}
            Whether or not the ratingArray should be shuffeled. (False)
        nmfflag : {True, False}
            Whether or not the factors should be non-negative.

        Returns
        -------
        describe : RSVD
            The trained model. 

        Note
        ----
        It is assumed, that the `ratingsArray` is proper shuffeld. 
        If the randomize flag is set the `ratingArray` is shuffeled every 10th
        epoch. 

        """
        model=RSVD() # make the instance of RSVD
        # set the num_movies, num_users that is the first and the second of dims
        model.num_movies,model.num_users=dims

        model.factors=factors
        model.lr=learnRate
        model.nmfflag = nmfflag
        # reg is boolean value that shows the usage of regularization
        model.reg=regularization
        model.min_improvement=minImprovement
        model.max_epochs=maxEpochs

        # convert ratings to float64 due to numerical problems
        # when summing over a huge number of values 
        avgRating = float(ratingsArray['f2'].astype(np.float64).sum()) / \
                    float(ratingsArray.shape[0])

        model.min_rating=ratingsArray['f2'].min()
        model.max_rating=ratingsArray['f2'].max()

        # initial value -- 
        # TODO: is it suitable?
        initVal=np.sqrt(avgRating/factors)

        rs=np.random.RandomState(randseed)
        
        # define the movie factors U
        # initilize by uniformally random variables
        if nmfflag:
            model.u=rs.uniform(\
                0.,randomNoise, model.num_movies*model.factors)\
                .reshape(model.num_movies,model.factors)+initVal

        else:
            model.u=rs.uniform(\
                -randomNoise,randomNoise, model.num_movies*model.factors)\
                .reshape(model.num_movies,model.factors)+initVal
        
        # define the user factors V
        # initilize by uniformally random variables
        if nmfflag:
            model.v=rs.uniform(\
                0.,randomNoise, model.num_users*model.factors)\
                .reshape(model.num_users,model.factors)+initVal

        else:
            model.v=rs.uniform(\
                -randomNoise,randomNoise, model.num_users*model.factors)\
                .reshape(model.num_users,model.factors)+initVal

        # receive a similarity matrix
        model.simtx = simtx
        model.gamma = gamma
        # finish the initialization by this line
        # start to training
        __trainModel(model,ratingsArray,probeArray,randomize=randomize)
        return model


def __trainModel(model,ratingsArray,probeArray,randomize=False):
    """Trains the model on the given rating data.
    
    If `probeArray` is not None the error on the probe set is
    determined after each iteration and early stopping is done
    if the error on the probe set starts to increase.

    If `randomize` is True the `ratingsArray` is shuffled
    every 10th epoch.

    Parameters
    ----------
    model : RSVD
        The model to be trained.
    ratingsArray : ndarray
        The numpy record array holding the rating data.
    probeArray : ndarray
        The numpy record array holding the validation data.
    out : file
        File to which debug msg should be written. (default stdout)
    randomize : {True,False}
        Whether or not the training data should be shuffeled every
        10th iteration. 

    Notes
    -----
    * Shuffling may take a while.
    
    """
    # redefine the parameters by the efficient representation on cython
    cdef object[Rating] ratings=ratingsArray
    early_stopping=False
    cdef object[Rating] probeRatings=probeArray
    if probeArray is not None:
        early_stopping=True
    cdef int n=ratings.shape[0]
    cdef int nMovies=model.num_movies### number of movies
    cdef int nUsers=model.num_users###
    cdef int i,k,epoch=0
    cdef int K=model.factors
    cdef int max_epochs=model.max_epochs
    cdef np.uint16_t m=0
    cdef np.uint32_t u=0
    cdef np.double_t uTemp,vTemp,err,trainErr
    cdef np.double_t lr=model.lr
    cdef np.double_t reg=model.reg
    cdef double probeErr=0.0, oldProbeErr=0.0
    cdef double min_improvement = model.min_improvement
    cdef char nflag = 0
    if model.nmfflag:
        nflag = 1
    
    cdef np.ndarray U=model.u   
    cdef np.ndarray V=model.v
    cdef np.ndarray SIMTX=model.simtx

    cdef double gamma = model.gamma
    
    cdef double *dataU=<double *>U.data
    cdef double *dataV=<double *>V.data

    cdef double *simtx = <double *>SIMTX.data
    
    
    print("########################################")
    print("             Factorizing                ")
    print("########################################")
    print("factors=%d, epochs=%d, lr=%f, reg=%f, n=%d, nmf=%d, gamma=%f" % (K,max_epochs,lr,reg,n,nflag,gamma))
    sys.stdout.flush()
    if early_stopping:
        oldProbeErr=probe(<Rating *>&(probeRatings[0]),\
                          dataU,dataV,K,probeRatings.shape[0])
        print("Init PRMSE: %f" % oldProbeErr)
        sys.stdout.flush()

    trainErr=probe(<Rating *>&(ratings[0]), dataU, dataV,K,n)
    trainerrlist = []
    seqreglist = []

    print("Init TRMSE: %f" % trainErr)
    print("----------------------------------------")
    print("epoche\ttrain err\tprobe err\telapsed time")
    sys.stdout.flush()
    # This line is based on the old style of cython and pyrex
    # 'epoch' is the parameter representing the times svd ran
    for epoch from 0 <= epoch < max_epochs:
        t1=time()
        if randomize and epoch%10==0:
            print("Shuffling training data\t")
            sys.stdout.flush()
            np.random.shuffle(ratings)
            print("done")

        # Calculate the dataU and dataV on this epoch
        trainErr=train(<Rating *>&(ratings[0]), dataU, \
                            dataV, K,n, reg, lr, nflag,
                            simtx, nMovies, gamma)

        if early_stopping:
            probeErr=probe(<Rating *>&(probeRatings[0]),dataU, \
                                dataV,K,probeRatings.shape[0])
            if oldProbeErr-probeErr < min_improvement:
                print("Early stopping\nRelative improvement %f" \
                          % (oldProbeErr-probeErr))
                break
            oldProbeErr = probeErr
        print("%d\t%f\t%f\t%f"%(epoch,trainErr,probeErr,time()-t1))

        # calc sequencial regularization value
        seqreg = 0.
        for reg_i from 0 <= reg_i < nMovies:
            for reg_j from 0 <= reg_j < nMovies:
                seqreg += simtx[movie*factors + reg_j] *\
                   (dataU[movie*factors + k] - dataU[j * factors + k])

        """
        seqreg = 0.
        for reg_i in range(len(U.data)):
            for reg_j in range(len(U.data)):
                for reg_k in range(len(U.data[0]))

            seqreg = seqreg + simtx[i,j]*

        # record progress
        trainerrlist.append(trainErr)
        seqreglist.append(seqreg)
        """
        sys.stdout.flush()

# The Rating struct. 
cdef struct Rating:
    np.uint16_t movieID
    np.uint32_t userID
    np.float32_t rating


cdef double predict(int uOffset,int vOffset, \
                        double *dataU, double *dataV, \
                        int factors):
    """Predict the rating of user i and movie j by first computing the
    dot product of the user and movie factors. 
    """
    cdef double pred=0.0
    cdef int k=0
    for k from 0<=k<factors:
        pred+=dataU[uOffset+k] * dataV[vOffset+k]
    return pred


cdef double simreg(double *dataU, int k, int factors, int movie, double *simtx, int nMovies):
    """
    Return: gradient value of the similarity based regularization.
    """
    cdef int j = 0
    cdef double reg = 0.0
    """
    CAUTION:
    Codes for exponential, and other regularization of simtx
    moved into himf.py.
    """
    for j from 0 <= j < nMovies:
        reg += simtx[movie*factors + j] *\
               (dataU[movie*factors + k] - dataU[j * factors + k])

    return reg
"""
def simreg_primitive(double *dataU, int k, int factors, int movie, double *simtx, int nMovies):
    for i in range(nMovies):
        for j in range(nMovies):
            prim += math.exp(simtx[i*factors + j])
"""

cdef double train(Rating *ratings, \
                            double *dataU, double *dataV, \
                            int factors, int n, double reg,double lr, char nflag, \
                            double *simtx, int nMovies, double gamma):
    """The inner loop of the factorization procedure.

    Iterate through the rating array: for each rating compute
    the gradient with respect to the current parameters
    and update the movie and user factors, resp. 
        factors: K, the number of dimension of the u and v factors
    """
    cdef int k=0,i=0,uOffset=0,vOffset=0
    cdef int user=0
    cdef int movie=0
    cdef Rating r
    cdef double uTemp=0.0,vTemp=0.0,err=0.0,sumSqErr=0.0

    for i from 0<=i<n:
        # calculate the error
        r=ratings[i]
        user=r.userID
        movie=r.movieID-1
        if movie < 0 or movie >= nMovies:
            print "movie range error", movie, nMovies

        uOffset=movie*factors #### debug point
        vOffset=user*factors
        err=<double>r.rating - \
            predict(uOffset,vOffset, dataU, dataV, factors)
        sumSqErr+=err*err;

        # calculate the new U and V
        for k from 0<=k<factors:
            uTemp = dataU[uOffset+k]
            vTemp = dataV[vOffset+k]

            """
            BEGIN: gradient test code
            """
#            if k == 0:
 #               print(simreg(dataU, k, factors, movie, simtx, nMovies))

            """
            END: gradient test code
            """

            if nflag:
                dataU[uOffset+k] = max(0., dataU[uOffset+k]+lr*(err*vTemp-reg*uTemp
                    - gamma*simreg(dataU, k, factors, movie, simtx, nMovies)))
                dataV[vOffset+k] = max(0., dataV[vOffset+k]+lr*(err*uTemp-reg*vTemp))
            else:
                dataU[uOffset+k] = dataU[uOffset+k]+lr*(err*vTemp-reg*uTemp
                    - gamma*simreg(dataU, k, factors, movie, simtx, nMovies))
                dataV[vOffset+k] = dataV[vOffset+k]+lr*(err*uTemp-reg*vTemp)
    # Record the progress


    return np.sqrt(sumSqErr/n)


cdef double probe(Rating *probeRatings, double *dataU, \
                      double *dataV, int factors, int numRatings):
    cdef int i,uOffset,vOffset
    cdef unsigned int user
    cdef unsigned short movie
    cdef Rating r
    cdef double pred = 0.0
    cdef double err,sumSqErr=0.0
    for i from 0<=i<numRatings:
        r=probeRatings[i]
        user=r.userID
        movie=r.movieID-1
        uOffset=movie*factors
        vOffset=user*factors
        pred = predict(uOffset,vOffset, dataU,dataV,factors)
        #if np.isnan(pred):
        #    print "pred is nan, i=%d, doc=%d, term=%d" % (i,movie,user)
        err=(<double>r.rating) - pred
        #if np.isnan(err):
        #    print "err is nan, i=%d, doc=%d, term=%d" % (i,movie,user)
        
        sumSqErr+=err*err
        #if i % 1000000 == 0.0:
        #    print err*err, sumSqErr, numRatings, np.sqrt(sumSqErr/numRatings)
    return np.sqrt(sumSqErr/numRatings)
