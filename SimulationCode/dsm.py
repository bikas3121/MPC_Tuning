# %%
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math
# from my_functions import direct_quantizer
from DirectQantization import quantise_signal

from DirectQantization import quantise_signal, generate_code, generate_dac_output
from  quantiser_configurations import quantiser_configurations, get_measured_levels
 # %%



# %% # Delta_sigma quantizer

# def dsm(Xcs, Qstep, QMODEL, YQns, MLns):
#     YQns = YQns.squeeze()
#     MLns = MLns.squeeze()


#     C = np.zeros((1, Xcs.size)).astype(int)  # save codes

#     # initialize
#     Xcs_dsm_1 =  np.array([np.floor(Xcs[0]/Qstep +1/2)*Qstep])
#     Xcs_dsm = np.array([Xcs_dsm_1])
#     E_dsm = np.array([Xcs[0] - Xcs_dsm_1[0]])
#     C[0,0] = Xcs_dsm_1
    
#     for i in range(1, len(Xcs)):
#         v_i = Xcs[i] - E_dsm[i-1]
#         qd_i = math.floor(v_i/Qstep +1/2)
#         Vmin = np.min(YQns)
#         c_i = qd_i - math.floor(Vmin/Qstep)

#         # Stauration limit
#         if c_i >= np.max(YQns):
#             c_i = np.max(YQns)
#         elif c_i <= np.min(YQns):
#             c_i = np.min(YQns)

#         C[0, i] = c_i #save codes
#         match QMODEL:
#             case 1:
#                 y_i = YQns[c_i]
#             case 2:
#                 y_i = MLns[c_i]
#         # quantise_signal(v_i, Qstep, Qlevels, Qtype)
#         q_i = y_i - v_i 
#         Xcs_dsm = np.append(Xcs_dsm, y_i)
#         E_dsm = np.append(E_dsm, q_i)
#     return C.astype(int)

def dsm(Nb, Xcs, Qstep, QMODEL, YQns, MLns):
    YQns = YQns.squeeze()
    MLns = MLns.squeeze()
    C_DSM = np.zeros((1, Xcs.size)).astype(int)     # save codes
    yns =  np.zeros((1,1))
    e = np.zeros((1,1))
    for i in range(0, Xcs.size):
        w = Xcs[i]
        v = w - e[0]

        # Re-quantizer (mid-tread)
        q = math.floor(v/Qstep + 0.5)  # quantize
        Vmin = np.min(YQns)
        c = q - math.floor(Vmin/Qstep)  # code

        # Stauration limit
        if c > 2**Nb - 1:
            c = 2**Nb - 1
        if c < 0:
            c = 0

        C_DSM[0, i] = c #save codes

        match QMODEL:
            case 1:
                y = YQns[c]
            case 2:
                y = MLns[c]
        e[0] = y - v
    return C_DSM.astype(int)
 

if __name__ == "__main__":
    # Sampling rate
    Fs = 100000;  # sampling frequency
    Ts = 1/Fs; # sampling rate
    t_end = 0.2; # time vector duration
    t = np.arange(0,t_end,Ts)  # time vector


    Qconfig = 2
    Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(Qconfig)
    # Reference signal
    Xcs_FREQ = 9  # reference singal's frequency
    Xcs = 3*np.sin(2*np.pi*Xcs_FREQ*t) + 3 # ref signal

    # %% # Quatnizer set

    YQns = YQ

    MLns = get_measured_levels(2).squeeze()
    #  %% Reconstrunction filter 
    Fc = 6000 # cutooff frequency
    Wn = Fc / (Fs / 2)
    b, a = signal.butter(2, Wn, 'low')
    w, h = signal.freqs(b, a)  #w - angular frequency, h= frequency response

    # %%
    # Direct quantizer
    Xcs_direct = quantise_signal(Xcs, Qstep, YQ, Qtype)  # directly quantized vectors 

    fig, ax  = plt.subplots()
    ax.plot(t, Xcs)
    ax.plot(t,Xcs_direct)
    plt.show()

    QMODEL = 2
    # C_DSM = np.zeros((1, Xcs.size)).astype(int)     # save codes
    # yns =  np.zeros((1,1))
    # e = np.zeros((1,1))
    # for i in range(0, Xcs.size):
    #     w = Xcs[i]
    #     v = w - e[0]

    #     # Re-quantizer (mid-tread)
    #     q = math.floor(v/Qstep + 0.5)  # quantize
    #     Vmin = np.min(YQns)
    #     c = q - math.floor(Vmin/Qstep)  # code

    #     # Stauration limit
    #     if c >= np.max(YQns):
    #         c = np.max(YQns)
    #     elif c <= np.min(YQns):
    #         c = np.min(YQns)

    #     C_DSM[0, i] = c #save codes

    #     match QMODEL:
    #         case 1:
    #             y = YQns[c]
    #         case 2:
    #             y = MLns[c]
    #     e[0] = y - v
 
    C_DSM = dsm(Nb, Xcs, Qstep, QMODEL, YQns, MLns)
    match QMODEL:

        case 1:
            Xcs_DSM = generate_dac_output(C_DSM, YQns).squeeze()
        case 2:
            Xcs_DSM = generate_dac_output(C_DSM, MLns).squeeze()
    # %% Filtering 
    Xcs_filt = signal.filtfilt(b, a, Xcs) # filter reference signal
    Xcs_direct_filt =signal.filtfilt(b, a, Xcs_direct)
    Xcs_delta_filt = signal.filtfilt(b, a, Xcs_DSM)

    plt.figure()
    plt.plot(t, Xcs_filt)
    plt.plot(t,Xcs_direct_filt)
    plt.plot(t,Xcs_delta_filt)
    plt.legend(['Reference','Direct','DSM'])
    plt.xlabel('Time')
    plt.title('Filtered Signals')
    plt.grid(True)



# %%
