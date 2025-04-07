
import numpy as np
import scipy.signal as signal

def exitwitherror(message='No error message', status=99):        
    import sys
    print(message, file=sys.stderr, flush=True)
    sys.exit(status)

def get_signal(l, signal_mode='sweep'):
    """
    Synthesize a test signal of length l. "mode" may be one of "sweep",
    "noise", "tones", "high", "exp"
    """

    B = np.zeros(l)
    
    if signal_mode=='sweep':
        
        f0 = 0
        f1 = l
        
        phases = np.power(np.linspace(np.sqrt(f0/2),np.sqrt(f1/2),l),2)        
        B[:l] = np.sin(phases*np.pi*2)        
        B[:l] *= signal.windows.gaussian(l,l/4)
        B+=np.random.rand(len(B))/len(B)*4

    elif signal_mode=='noise':

        FB = rfft(B)
        FB[0]=1
        FB[1:-1] = np.exp( 2j * np.pi * np.random.rand( len(FB) - 2 ))
        FB[1:-1] = fftconvolve(FB[1:-1],
                               signal.windows.gaussian(len(FB),32),
                               'same')
        B = irfft(FB)
        
    elif signal_mode=='tones':

        f0 = 70 / 24000 * l
        f1 = f0 * np.exp( 1 )
        
        B =  np.sin( np.linspace( 0, 2*np.pi*f0, l ) )
        B += np.sin( np.linspace( 0, 2*np.pi*f1, l ) )
            
        B *= signal.windows.gaussian(l, l/12)
        
        B = np.roll(B, len(B)//2)
        
    elif signal_mode=='high':

        f0 = 0.00125 
        order = 4
        
        B[0] = 1
        b,a = signal.butter(order,f0,'high')

        CB = copy(B)
        B = np.zeros(l)

        d = 0
        maxd = int(np.sqrt(l))
        
        for i in range(2,maxd):
             B += np.roll(CB,d) / i
             d += maxd-i
                            
        B *= signal.windows.gaussian(l, l/2)

    elif signal_mode=='exp':

        f0 = 0.00125 
        
        B = np.exp(np.linspace(0,-f0*l,len(B)))
        B *= signal.windows.gaussian(l, l/2)

        B[0]*=-1
        

    else:

        exitwitherror(f"Unknown signal mode {signal_mode}")

    return B
