import numpy as np

from matplotlib.pyplot import *

from numpy.fft import fft,ifft,rfft,irfft
from copy import copy

from scipy.io.wavfile import read,write
from scipy.signal import fftconvolve, hilbert

import scipy.signal as signal

from measure_shell import measure
from signals import get_signal

# initialize pyplot for convenience

ion()
show()

def exitwitherror(message='No error message', status=99):        
    import sys
    print(message, file=sys.stderr, flush=True)
    sys.exit(status)

from analysis import analysis

if True:    #  space for loops etc.
    
    k = pow(2,20)

    # Two Golay-like sequences
    #
    # The sum of the autocorrelations of these signals is nearly equal to
    #  a unit pulse. This property is used later.
    #
    # I had my computer search blindly for such sequences, the first one
    # is 7 samples long by 12 phases, the second one is 7x5 

    G0 = np.array([[ 1.       +0.00000000e+00j,  0.8660254-5.00000000e-01j],
                   [-0.8660254+5.00000000e-01j, -0.8660254+5.00000000e-01j],
                   [ 1.       +0.00000000e+00j,  0.5      -8.66025404e-01j],
                   [-1.       +1.22464680e-16j, -0.8660254+5.00000000e-01j],
                   [ 0.5      +8.66025404e-01j,  0.8660254-5.00000000e-01j],
                   [-0.8660254-5.00000000e-01j, -0.8660254-5.00000000e-01j],
                   [-1.       +1.22464680e-16j,  1.       +0.00000000e+00j]])

    G1 = np.array([[-0.5      +0.8660254j,  0.8660254+0.5j      ],
                   [ 0.8660254-0.5j      , -0.5      +0.8660254j],
                   [ 0.5      +0.8660254j,  0.8660254-0.5j      ],
                   [ 0.5      +0.8660254j,  0.8660254+0.5j      ],
                   [-0.5      -0.8660254j,  0.8660254+0.5j      ],
                   [ 0.5      +0.8660254j, -0.8660254-0.5j      ],
                   [ 0.5      +0.8660254j,  0.8660254+0.5j      ]])

    # "Convolve" the sequences
        
    H = copy(G0)

    for i in range(1):
        if i%2==0:
            G_ = copy(G1)
        else:
            G_ = copy(G0)
        H_ = np.zeros((len(G_)*len(H),2),'complex')
        for j in range(len(H)):
            for kk in range(len(G_)):
                H_[j*len(G_)+kk,:]=H[j,:]*G_[kk,:]
        H = copy(H_)
                
    G = copy(H_)

    print(f"Sequence length: {len(G)}")

    print("Synthesizing test signal")

    # G contains a pair of near-complementary sequences
    
    Seq1=G[:,0]
    Seq2=G[:,1]

    # Initialize test signal as a realization of the signal

    A1 = np.zeros(k,'complex')
    A2 = np.zeros(k,'complex')

    u = k//len(Seq1)

    # We don't want to output 180° phase difference
    
    a = 0.8
    b = 1-a

    A1[:u*len(Seq1):u]=Seq1*a+Seq2*b
    A2[:u*len(Seq2):u]=Seq2*a+Seq1*b

    A1 /= np.linalg.norm(A1,1)
    A2 /= np.linalg.norm(A2,1)

    # Synthesize a mono test signal

    l = u
    
    signal_mode='exp'

    print(f"Signal mode: {signal_mode}")

    B = get_signal(l, signal_mode)
    
    OB = copy(B)
            
    A1_ = np.zeros(k)
    A2_ = np.zeros(k)

    A1_[:len(B)]=B
    A2_[:len(B)]=B

    # convolve the realized sequences and the test signal via fft, realize
    # the signals
    
    A1 = np.real( ifft( fft(A1) * fft(A1_) ) )
    A2 = np.real( ifft( fft(A2) * fft(A2_) ) )

    # keep the original signals
    
    OA1 = copy(A1)
    OA2 = copy(A2)

    # modify the signal slightly

    FA1 = rfft(A1)
    FA2 = rfft(A2)

    FA1 /= signal.windows.gaussian(len(FA1),len(FA1)/3*2)
    FA2 /= signal.windows.gaussian(len(FA2),len(FA2)/3*2)
    FA1 *= np.exp(1j*np.linspace(0,12,len(FA1)))
    FA2 *= np.exp(1j*np.linspace(0,12,len(FA2)))

    A1 = irfft(FA1)
    A2 = irfft(FA2)
    
    A = np.vstack([A1,A2])

    # pad for output

    AF = np.hstack([np.zeros((2,16384*3)),A,np.zeros((2,16384*3))])
    AF /= np.max(AF,axis=1)[None,:].T
    AF *= 0.999

    
    print("Measuring")
    
    result = measure( AF )
    
    
    # Pad the measured signal

    B = result
    
    pad = np.zeros(((len(B)-len(OA1))//2,2))

    A = np.vstack([pad,np.vstack([OA1,OA2]).T,pad])

    # padding
    
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

    # Mute the supposedly silent part of the received signal 

    _B[ :16384*3-1024  ] = 0
    _B[ -16384*3+1024: ] = 0

    print("Transforming")

    l = len(A)
    
    W1 = signal.windows.gaussian(l*2,l*0.5)[-l:]
    W2 = signal.windows.gaussian(l*2,l*0.5)[-l:]

    FA = rfft( A  * W1[:,None] , axis=0)
    FB = rfft( B_ * W2 , axis=0)
    
    FAe = rfft( ( A  + np.flip(A, axis=0 ) )* W1[:,None] , axis=0)
    FBe = rfft( ( B_ + np.flip(B, axis=0 ) )* W2 , axis=0)

    FSe = np.abs(FBe[:,None]) / ( 1e-16 + np.abs(FAe) )

    FAo = rfft( ( A  - np.flip(A, axis=0 ) )* W1[:,None] , axis=0)
    FBo = rfft( ( B_ - np.flip(B, axis=0 ) )* W2 , axis=0)
    
    FSo = np.abs(FBe[:,None]) / ( 1e-16 + np.abs(FAo) )

    # divide signals, smooth and subsample
        
    ILFSo=analysis(FAo,FBo,14) 
    ILFSe=analysis(FAe,FBe,14)
    ILFS=analysis(FA,FB,14) 

    Se=irfft( np.exp( hilbert( -ILFSe, axis=0) ), axis=0 )
    So=irfft( np.exp( hilbert( -ILFSo, axis=0) ), axis=0 )

    S = Se + So

    T = np.roll( np.flip( S, axis=0 ), 32, axis=0 )

    clf()
    
    plot( T )

    # Apply a window of 1024
    # We're ignoring any accidential imaginary part

    X = copy( np.real(T).astype('float') ) [:1024, :]

    # convert to signed int
    
    Q = np.array( X / np.max(abs(X)) * 32760, 'int16') 

    # write a file
    
    write( "equalizer.wav", 48000, Q )
