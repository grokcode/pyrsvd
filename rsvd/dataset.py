"""
Datasets
--------

Netflix
-------
Currently, only pyflix converter implemented. 



MovieLens
---------
Parses MovieLens data 


"""

# imports from python stdlib
import sys

from os.path import exists
# non python stdlib imports

import numpy as np

from rsvd import rating_t
    

# File Metadata
__version__="0.1"
__author__="peter.prettenhofer@gmail.com"
__license__="mit"

class Dataset(object):
    """Dataset interface
    """

    def __init__(self,movieIDs,userIDs,ratings):
        self._movieIDs=movieIDs
        self._userIDs=userIDs
        self._ratings=ratings

    def userIDs(self):
        return self._userIDs

    def movieIDs(self):
        return self._movieIDs

    def ratings(self):
        return self._ratings

    def rmse(self,model):
        """Compute the RMSE of the given model w.r.t. the ratings. 
        """
        sqerr=0.0
        for movieID,userID,rating in self._ratings:
            err=rating-model(movieID,userID)
            sqerr+=err*err
        return np.sqrt(sqerr/self._ratings.shape[0])

class MovieLensDataset(Dataset):
    """The MovieLens dataset.

    There exist three different versions of the dataset:
       1. asd
       2. af
    see U{GroupLens <http://www.grouplens.org/>} for further information
    """

    def __init__(self, movieIDs,userIDs,ratings):
        """
        """
        super(MovieLensDataset,self).__init__(movieIDs,userIDs,ratings)


    @classmethod
    def loadDat(cls,file):
        """Loads the MovieLens dataset via the ratings.dat file. 

        """
        if not exists(file):
            raise ValueError("%s file does not exist" % file)
        f=open(file)
        try:
            rows=[tuple(map(int,l.rstrip().split("::"))) for l in f.readlines()]
            n=len(rows)
            
            # define rating array (itemID,userID,rating)
            ratings=np.empty((n,),dtype=rating_t)
            for i,row in enumerate(rows):
                ratings[i]=(row[1],row[0]-1,row[2])
            movieIDs=np.unique(ratings['f0'])
            userIDs=np.unique(ratings['f1'])
            
            movieIDs.sort()
            userIDs.sort()
            #map movieIDs
            for i,rec in enumerate(ratings):
                ratings[i]['f0']=movieIDs.searchsorted(rec['f0'])+1

            #original_movieIDs=movieIDs
                
            movieIDs=np.unique(ratings['f0'])
            movieIDs.sort()
            return MovieLensDataset(movieIDs,userIDs,ratings)
        finally:
            f.close()


class EloChessDataset(Dataset):
    """The Elo-Chess dataset.

    whitePlayer corresponds to movieId.
    blackPlayer to userId.

    see U{GroupLens <http://www.grouplens.org/>} for further information
    """

    def __init__(self, movieIDs,userIDs,ratings):
        """
        """
        super(EloChessDataset,self).__init__(movieIDs,userIDs,ratings)


    @classmethod
    def loadCSV(cls,file):
        """Loads the dataset via a csv file file. 
        """
        if not exists(file):
            raise ValueError("%s file does not exist" % file)
        f=open(file)
        try:
	    f.readline() # discard file header
            lineit = (l.split(",") for l in f.readlines())
	    rows = [(int(l[1]),int(l[2]),float("0"+l[3])) for l in lineit]
            n=len(rows)
            
            # define rating array (itemID,userID,rating)
            ratings=np.empty((n,),dtype=rating_t)
            for i,row in enumerate(rows):
                ratings[i]=(row[0],row[1],row[2])
            movieIDs=np.unique(ratings['f0'])
            userIDs=np.unique(ratings['f1'])
            
            movieIDs.sort()
            userIDs.sort()            
            
            return EloChessDataset(movieIDs,userIDs,ratings)
        finally:
            f.close()



class NetflixDataset(Dataset):
    """
    """

    def __init__(self, movieIDs,userIDs,ratings):
        """
        """
        super(NetflixDataset,self).__init__(movieIDs,userIDs,ratings)


    @classmethod
    def loadArray(cls,file):
        """Loads the Netflix dataset from a serialized numpy record array. 
        
        """        
        f=open(file,'r')
        try:
            ratings=np.fromfile(f,dtype=rating_t)
            movieIDs=np.unique(ratings['f0'])
            userIDs=np.unique(ratings['f1'])
            return NetflixDataset(movieIDs,userIDs,ratings)
        finally:
            f.close()

def main(prog_args):
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv) or 0)


