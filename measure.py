import numpy as np

from matplotlib.pyplot import *

from numpy.fft import fft,ifft,rfft,irfft
from copy import copy

from scipy.io.wavfile import read,write
from scipy.signal import fftconvolve, hilbert

import scipy.signal as signal

from measure_shell import measure

# initialize pyplot for convenience

ion()
show()


if True:    #  blank
    
    k = pow(2,20)


    G = np.array([[2,14],
                  [14,-2]])

    # use a Golay primitive as the first root to get stereo straight
    
    H = np.array([[1,1],[1,-1]]) 
    
    for i in range(10):
        H_ = np.zeros((len(G)*len(H),2))
        for j in range(len(H)):
            for kk in range(len(G)):
                H_[j*len(G)+kk,:]=H[j,:]*G[kk,:]
        H = copy(H_)
                
    G = copy(H_)

    # G now contains a pair of near-complementary sequences, i.e. the sum
    # of the auto-correlations of both channels is close to a unit pulse
    
    Seq1=G[:,0]
    Seq2=G[:,1]

    # Initialize test signal as a realization of the signal

    A1 = np.zeros(k)
    A2 = np.zeros(k)

    u = k//len(Seq1)

    A1[:u*len(Seq1):u]=Seq1
    A2[:u*len(Seq2):u]=Seq2 


    # Synthesize a pulse signal
    
    B = np.zeros(16384)

    # We'll take the difference of two sine waves

    B = np.sin(np.linspace(0,np.pi*2*0.0004,len(B)))
    B -= np.sin(np.linspace(0,np.pi*2*0.0005,len(B)))
    
    l = len(B)//601 # approximately 40 Hz in 48 kHz sampling, a deep bass
                    # also a prime number

    # repeat the test signal a few times, getting softer

    OB = copy(B)

    q = 601

    for i in range(1,len(B)//q//2):
        B+=np.roll(OB,i*q)/pow(1.5,i) 

    # apply some filters

    b,a = signal.butter(1,80/24000,'high')
    B = signal.lfilter(b,a,B)[:len(B)]
    b,a = signal.butter(1,0.5,'low')
    B = signal.lfilter(b,a,B)[:len(B)]
    
        
    A1_ = np.zeros(k)
    A2_ = np.zeros(k)

    A1_[:len(B)]=B
    A2_[:len(B)]=B

    # convolve the realized sequence and the test signal via fft
    
    A1 = irfft(rfft(A1)*rfft(A1_))
    A2 = irfft(rfft(A2)*rfft(A2_))

    # apply a bandpass around fs/2

    FA1 = rfft(A1)
    FA2 = rfft(A2)

    FA1 *= FA1*signal.windows.gaussian(len(FA1),len(FA1)/4)
    FA2 *= FA1*signal.windows.gaussian(len(FA1),len(FA1)/4)

    # shift the phases a little

    R1 = fftconvolve(np.random.rand(len(FA1))-0.5,signal.windows.gaussian(len(FA1),len(FA1)/8),'same')
    R2 = fftconvolve(np.random.rand(len(FA1))-0.5,signal.windows.gaussian(len(FA1),len(FA1)/16),'same')

    FA1 *= np.exp(1j*(R1+R2)*0.001)
    FA2 *= np.exp(1j*(R1-R2)*0.001)

    A1 = irfft(FA1)
    A2 = irfft(FA2)

    # keep the original signals
    
    OA1 = copy(A1)
    OA2 = copy(A2)
    
    l = len(A1)//2+1

    A = np.vstack([A1,A2])

    # pad for output

    AF = np.hstack([np.zeros((2,16384*3)),A,np.zeros((2,16384*3))])
    AF /= np.linalg.norm(AF,1)
    AF *= 0.999
    
    
    print("Measuring")
    
    result = measure( AF )
    
    
    # Pad the measured signal

    B = result

    f0 = 20
    f1 = 100
    
    pad = np.zeros(((len(B)-len(OA1))//2,2))

    A = np.vstack([pad,np.vstack([OA1,OA2]).T,pad])

    # padding, in case
    
    A_len = A.shape[0]
    B_len = B.shape[0]

    if A_len > B_len:
        B_ = np.concatenate([B, np.zeros(A_len - B_len)])  
    elif A_len < B_len:
        B_ = B[:A_len]
    else:
        B_ = B


    
    _A = copy(A)
    _B = copy(B_)

    _B[ :16384*3-1024  ] = 0
    _B[ -16384*3+1024: ] = 0

    print("Transforming")

    FA = rfft( A  , axis=0)
    FB = rfft( B_ , axis=0)
    
    # compute signal spectrum
    
    FS = np.abs(FB[:,None]) / ( 1e-16 + np.abs(FA) )
    
    k = pow(2,16)

    # smooth the spectrum

    FS_ = fftconvolve(abs(FS),
                      np.ones(k)[:,None],
                      'same',axes=0)

    LFS = np.log(1e-16 + abs(FS_))

    l = FA.shape[0]

    k = pow(2,16)

    LFS_ = LFS / np.max(LFS)

    ILFS = LFS_[ ::len(LFS_)//pow(2,10) ] # take 1024 samples

    # retransform

    FT = np.exp( hilbert( -ILFS , axis=0 ) )

    T = irfft( FT , axis=0 )

    write("equalizer.wav", 48000, T )

