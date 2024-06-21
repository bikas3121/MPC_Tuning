#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""  Run DAC simulation using various noise shaping techniques

@author: Bikash Adhikari 
@date: 23.02.2024
@license: BSD 3-Clause
"""

# %%
import numpy as np
from scipy  import signal
import scipy
import csv
import matplotlib.pyplot as plt
import statistics

from DirectQantization import quantise_signal, generate_code, generate_dac_output
from  quantiser_configurations import quantiser_configurations, get_measured_levels

from MHOQ import MHOQ
from nsdcal import nsdcal, noise_shaping
from dsm import dsm


from signalProcessing import signalProcessing as SP
from plot_variance import plot_variance
from process_sim_output import process_sim_output
from balreal import balreal
# Generate test signal
def test_signal(SCALE, MAXAMP, FREQ, Rng,  OFFSET, t):
    """
    Generate a test signal (carrier)

    Arguments
        SCALE - percentage of maximum amplitude
        MAXAMP - maximum amplitude
        FREQ - signal frequency in hertz
        OFFSET - signal offset
        t - time vector
    
    Returns
        x - sinusoidal test signal
    """
    return (SCALE/100)*MAXAMP*np.cos(2*np.pi*FREQ*t) + OFFSET  + 2**(Nb-1)

# %% Chose how to compute SINAD
class sinad_comp:
    CFIT = 1        # curve fitting
    FFT = 2         # fft based 

# Choose sinad computation method
SINAD_COMP_SEL = sinad_comp.CFIT
# SINAD_COMP_SEL = sinad_comp.FFT
# %% Quantiser configurations 
Qconfig = 5
Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(Qconfig)
# %% Sampling frequency and rate
Fs = 1e6
Ts = 1/Fs

# %% Output low-pass filter cutoff order and cutoff frequency
N_lp = 2
Fc_lp = 1e5 # cutoff frequency
# %% Carrier signal
Xcs_SCALE = 100
Xcs_FREQ = 999

# %% Generate time vector

match 1:
    case 1:  # specify duration as number of samples and find number of periods
        Nts = 1e5  # no. of time samples
        Np = np.ceil(Xcs_FREQ*Ts*Nts).astype(int) # no. of periods for carrier

    case 2:  # specify duration as number of periods of carrier
        Np = 50 # no. of periods for carrier
        
Npt = 1  # no. of carrier periods to use to account for transients
Np = Np + Npt

t_end = Np/Xcs_FREQ  # time vector duration
t = np.arange(0, t_end, Ts)  # time vector

# %% Generate carrier/test signal
SIGNAL_MAXAMP = Rng/2 - Qstep  # make headroom for noise dither (see below)
SIGNAL_OFFSET = -Qstep/2  # try to center given quantiser type
Xcs = test_signal(Xcs_SCALE, SIGNAL_MAXAMP, Xcs_FREQ, Rng,  SIGNAL_OFFSET, t)

# fig, ax = plt.subplots()
# ax.plot(t, Xcs)

# %% Reconstruction filter parameters 
match 1:
    case 1:
        Fc_lpd = Fc_lp
        Wn = Fc_lpd / (Fs / 2)
        # Butterworth Filter properties 
        b, a = signal.butter(N_lp, Wn, 'low')  # tf numerator and denominator
        butter = signal.dlti(*signal.butter(N_lp, Wn,'low'))
        w, h = signal.dimpulse(butter, n = 10)  #w - angular frequency, h= frequency response
        h = h[0]  

    case 2:  # % Perception filter - Goodwin Paper -  equivalent low pass filter
        b = np.array([1, 0.91, 0])
        a = np.array([1 , -1.335, 0.644])
        percep = signal.dlti(b, a)
        w, h = signal.dimpulse(percep, n = 10)  #w - angular frequency, h= frequency response
        h = h[0]  

A1, B1, C1, D1 = signal.tf2ss(b,a) # Transfer function to StateSpace
A, B, C, D = balreal(A1, B1, C1, D1)
e, v = np.linalg.eig(A)
# %% Quantiser levels : Ideal and Measured
# Ideal quantiser levels
YQns = YQ     
# Measured quantiser levels
MLns = get_measured_levels(Qconfig)


# %% Optimal NTF 
br_ontf = np.array([1.0000,  -1.7601,   1.1830,   -0.2781])
ar_ontf = np.array([1.0000,   -0.3538,    0.3666,   -0.0330])

# %% Optimal NSF
bf_onsf =  ar_ontf - br_ontf
af_onsf = ar_ontf
[AF_ONSF, BF_ONSF, CF_ONSF, DF_ONSF] = signal.tf2ss(bf_onsf, af_onsf)

#% corresponding low pass filter 
bh_onsf = ar_ontf
ah_onsf = br_ontf
[AH_ONSF, BH_ONSF, CH_ONSF, DH_ONSF] = signal.tf2ss(bh_onsf, ah_onsf)
[AH_ONSF, BH_ONSF, CH_ONSF, DH_ONSF] = balreal(AH_ONSF, BH_ONSF, CH_ONSF, DH_ONSF)
# %% LIN methods on/off
DIR_ON = True
DSM_ON = True
NSD_ON = True
MPC_ON = True

# %% Quatniser Model
# Quantiser model: 1 - Ideal , 2- Calibrated
QMODEL = 1          
# %% Direct Quantization 
if DIR_ON:
    C_DIR = quantise_signal(Xcs, Qstep, YQns, Qtype).astype(int)
    match QMODEL:
        case 1:
            Xcs_DIR = generate_dac_output(C_DIR, YQns)
        case 2:
            Xcs_DIR = generate_dac_output(C_DIR, MLns)

# %% Delta Sigma Modulator
if DSM_ON:
    C_DSM = dsm(Xcs, Qstep, YQns)
    match QMODEL:
        case 1:
            Xcs_DSM = generate_dac_output(C_DSM, YQns)
        case 2:
            Xcs_DSM = generate_dac_output(C_DSM, MLns)



# %% NSD CAL
if NSD_ON:
    match 2:
        case 1:
            AM = np.array([[0.0, 0.0], [1.0, 0.0]])
            BM = np.array([[2.0], [0.0]])
            CM = np.array([[1.0, -0.5]])
            DM = np.array([[0.0]])
        case 2:
            AM = AF_ONSF
            BM = BF_ONSF
            CM = CF_ONSF
            DM = DF_ONSF

    # C_NSD = noise_shaping(Nb, Xcs, bns, ans, Qstep, YQns, MLns, Vmin, QMODEL)  
    C_NSD = nsdcal(Xcs, YQns, MLns, Qstep, Vmin, Nb, QMODEL, AM, BM, CM, DM)
    match QMODEL:
        case 1:
            Xcs_NSD = generate_dac_output(C_NSD, YQns)
        case 2:
            Xcs_NSD = generate_dac_output(C_NSD, MLns) 


# %% MPC : Prediction horizon
N = 1
if MPC_ON:
    match 1:
        case 1:
            AH = A
            BH = B
            CH = C
            DH = D
        case 2:
            AH = AH_ONSF
            BH = BH_ONSF
            CH = CH_ONSF 
            DH = DH_ONSF 
    # #%% Numerical MPC: Solving MHOQ numerically using Gurobi MILP formulation 
    MHOQ = MHOQ(Nb, Qstep, QMODEL, A, B, C, D)

    # # Binary Formulation - For smaller number of bits only
    # # MHOQ = MHOQ_BIN(Nb, Qstep, QMODEL, A, B, C, D)
    # Get Codes
    C_MHOQ = MHOQ.get_codes(N, Xcs, YQns, MLns)
    match QMODEL:
        case 1:
            Xcs_MHOQ = generate_dac_output(C_MHOQ, YQns)
        case 2:

            Xcs_MHOQ = generate_dac_output(C_MHOQ, MLns)

# %% Signal Processing
tm = t[:Xcs.size - N]

TRANSOFF = np.floor(Npt*Fs/Xcs_FREQ).astype(int)  # remove transient effects from output
sp = SP(Xcs, b, a, TRANSOFF)

# Filterted reference
F_Xcs = sp.referenceFilter

# %% Variance Pots
if DIR_ON:
    F_Xcs_DIR, err_DIR, var_DIR = sp.signalFilter(Xcs_DIR)
    yd = Xcs_DIR[0,:len(tm)].squeeze()
    yd_avg, ENOB_M = process_sim_output(tm, yd, Fc_lp, Fs, N_lp, TRANSOFF, SINAD_COMP_SEL, True, 'linear')
if DSM_ON:
    F_Xcs_DSM, err_DSM, var_DSM = sp.signalFilter(Xcs_DSM)
    ydsm = Xcs_DSM[0,:len(tm)].squeeze()
    ydsm_avg, ENOB_M = process_sim_output(tm, ydsm, Fc_lp, Fs, N_lp, TRANSOFF, SINAD_COMP_SEL, True, 'linear')
if NSD_ON:
    F_Xcs_NSD, err_NSD, var_NSD = sp.signalFilter(Xcs_NSD)
    yn = Xcs_NSD[0,:len(tm)].squeeze()
    yn_avg, ENOB_M = process_sim_output(tm, yn, Fc_lp, Fs, N_lp, TRANSOFF, SINAD_COMP_SEL, True, 'linear')
if MPC_ON:
    F_Xcs_MHOQ, err_MHOQ, var_MHOQ = sp.signalFilter(Xcs_MHOQ)
    ym = Xcs_MHOQ.squeeze()
    ym_avg, ENOB_M = process_sim_output(tm, ym, Fc_lp, Fs, N_lp, TRANSOFF, SINAD_COMP_SEL, True, 'linear')



# if DIR_ON and NSD_ON:
#     plot_variance(var_dir = var_DIR,  var_nsd = var_NSD)
    # fig, ax = plt.subplots()
    # ax.plot(t[TRANSOFF: TRANSOFF + len(F_Xcs_DIR)], F_Xcs.squeeze())
    # ax.plot(t[TRANSOFF: TRANSOFF + len(F_Xcs_DIR)], F_Xcs_DIR.squeeze())
    # ax.plot(t[TRANSOFF: TRANSOFF + len(F_Xcs_NSD)], F_Xcs_NSD.squeeze())
# if DIR_ON and NSD_ON and MPC_ON:
#     plot_variance(var_dir = var_DIR,  var_nsd = var_NSD, var_mhoq = var_MHOQ)
    # fig, ax = plt.subplots()
    # ax.plot(t[TRANSOFF: TRANSOFF + len(F_Xcs_DIR)], F_Xcs.squeeze())
    # ax.plot(t[TRANSOFF: TRANSOFF + len(F_Xcs_DIR)], F_Xcs_DIR.squeeze())
    # ax.plot(t[TRANSOFF: TRANSOFF + len(F_Xcs_NSD)], F_Xcs_NSD.squeeze())
    # ax.plot(t[TRANSOFF: TRANSOFF + len(F_Xcs_NSD)], F_Xcs_MHOQ.squeeze())

# %% Quantisation error
# q_mhoq = Xcs[0:Xcs_MHOQ.size] - Xcs_MHOQ.squeeze()
# fig, ax = plt.subplots()
# ax.plot(tm, q_mhoq)

""" with open("quant_err_butter_2nd_100k_fs10mhz_ideal.csv", 'w') as f1:
    wrt = csv.writer(f1, delimiter = '\t')
    wrt.writerow(q_mhoq) """

