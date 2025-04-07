
import numpy as np

from scipy.signal import fftconvolve, hilbert

def analysis(FA, FB, k = 12):
    #
    # Perform analysis, i.e. divide the fourier spectra, smoothen,
    # subsample, normalize
    #
    # 2^k is the 
    #

    FS = np.abs(FB[:,None]) / ( 1e-16 + np.abs(FA) )
        
    FS_ = fftconvolve(abs(FS),
                      np.ones(pow(2,k))[:,None],
                      'same',axes=0)

    LFS = np.log(1e-16 + abs(FS_))

    l = FA.shape[0]

    LFS_ = LFS / np.linalg.norm(LFS, 2, axis=0)

    ILFS = LFS_ [ ::len(LFS_) // pow(2,k) ]

    ILFS -= np.mean( ILFS, axis=0 )
    
    ILFS /= np.linalg.norm( ILFS, 2, axis=0 )

    return ILFS
