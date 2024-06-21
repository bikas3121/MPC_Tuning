# %%
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# from my_functions import direct_quantizer
from DirectQantization import quantise_signal
 # %%



# %% # Delta_sigma quantizer

def dsm(Xcs, Qstep, Qlevels ):

    Xcs_dsm_1 =  np.array([np.floor(Xcs[0]/Qstep +1/2)*Qstep])
    Xcs_dsm = np.array([Xcs_dsm_1])
    E_dsm = np.array([Xcs[0] - Xcs_dsm_1[0]])

    for i in range(1, len(Xcs)):
        v_i = Xcs[i] - E_dsm[i-1]
        y_i = np.floor(v_i/Qstep +1/2)*Qstep

        # Stauration limit
        if y_i >= np.max(Qlevels):
            y_i = np.max(Qlevels)
        elif y_i <= np.min(Qlevels):
            y_i = np.min(Qlevels)
        
        # quantise_signal(v_i, Qstep, Qlevels, Qtype)
        q_i = y_i - v_i 
        Xcs_dsm = np.append(Xcs_dsm, y_i)
        E_dsm = np.append(E_dsm, q_i)
    return Xcs_dsm.astype(int)



if __name__ == "__main__":
    # Sampling rate
    Fs = 100000;  # sampling frequency
    Ts = 1/Fs; # sampling rate
    t_end = 0.2; # time vector duration
    t = np.arange(0,t_end,Ts)  # time vector

    # Reference signal
    Xcs_FREQ = 9  # reference singal's frequency
    Xcs = 3*np.sin(2*np.pi*Xcs_FREQ*t)  # ref signal

    # %% # Quatnizer set
    Qlevels =  np.arange(-3, 4,1)
    Qstep = 1 
    Qtype = 1   # midtread

    #  %% Reconstrunction filter 
    Fc = 6000 # cutooff frequency
    Wn = Fc / (Fs / 2)
    b, a = signal.butter(2, Wn, 'low')
    w, h = signal.freqs(b, a)  #w - angular frequency, h= frequency response

    # %%
    # Direct quantizer
    Xcs_direct = quantise_signal(Xcs, Qstep, Qlevels, Qtype)  # directly quantized vectors 

    fig, ax  = plt.subplots()
    ax.plot(t, Xcs)
    ax.plot(t,Xcs_direct)
    plt.show()


    Xcs_dsm = dsm(Xcs, Qstep, Qlevels)
    # %% Filtering 
    Xcs_filt = signal.filtfilt(b, a, Xcs) # filter reference signal
    Xcs_direct_filt =signal.filtfilt(b, a, Xcs_direct)
    Xcs_delta_filt = signal.filtfilt(b, a, Xcs_dsm)

    plt.figure()
    plt.plot(t, Xcs_filt)
    plt.plot(t,Xcs_direct_filt)
    plt.plot(t,Xcs_delta_filt)
    plt.legend(['Reference','Direct','DSM'])
    plt.xlabel('Time')
    plt.title('Filtered Signals')
    plt.grid(True)


