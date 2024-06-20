#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Noise-shaping with digital calibration.

@author: Arnfinn Aas Eielsen
@date: 06.03.2024
@license: BSD 3-Clause
"""

import numpy as np
import math
from scipy import signal
from balreal import balreal

def nsdcal(X, YQns, MLns, Qstep, Vmin, Nb, QMODEL, bns, ans):
    """
    X
        input signal
    Dq
        re-quantiser dither
    YQns, 1d array
        ideal, uniform output levels (ideal model)
    MLns, 1d array
        measured, non-unform levels (calibration model)
    Qstep, Vmin, Nb
        quantiser params. (for re-quantisation and code generation)
    QMODEL
        choice of quantiser model
            1: ideal
            2: measured/calibrated
    """
    YQns = YQns.squeeze()
    MLns = MLns.squeeze()
    # Noise-shaping filter (using a simple double integrator)
    # b = np.array([1, -2, 1])
    # a = np.array([1, 0, 0])
    # Hns_tf = signal.TransferFunction(b, a, dt=1)  # double integrator
    # Mns_tf = signal.TransferFunction(a-b, a, dt=1)  # Mns = 1 - Hns
    # Mns = Mns_tf.to_ss()
    match 1:
        case 1:
            AM = np.array([[0.0, 0.0], [1.0, 0.0]])
            BM = np.array([[2.0], [0.0]])
            CM = np.array([[1.0, -0.5]])
            DM = np.array([[0.0]])
        case 2:  # Optimal NSF for 2nd Order butterworth filter
            AM = np.array([[-0.2376, -0.1816],[0.9999, 0.0004]])
            BM = np.array([[1.0],[0.0]])
            CM = np.array([[-1.3086, 0.2312]])
            DM= np.array([1.0])
        case 3:  # Optimal NSF for 3nd Order butterworth filter
            AM = np.array([[0.3538, -0.3666, 0.0330], [1.0000,  0,  0], [ 0, 1.0000 , 0]])
            BM = np.array([ [1], [0],[0]])
            CM = np.array([[-1.4063,    0.8164 ,  -0.2451]])
            DM = np.array([[0.0]])
    # Ad = AM 
    # Bd = BM
    # Cd = CM
    # Dd = DM
    # Ad = Mns.A
    # Bd = Mns.B
    # Cd = Mns.C
    # Dd = Mns.D
    # AM = np.array([[0.0, 0.0], [1.0, 0.0]])
    # BM = np.array([[2.0], [0.0]])
    # CM = np.array([[1.0, -0.5]])
    # DM = np.array([[0.0]])

    # Make a balanced realisation.
    # Less sensitivity to filter coefficients in the IIR implementation.
    # (Useful if having to used fixed-point implementation and/or if the filter order is to be high.)
    # Ad, Bd, Cd, Dd = balreal(Mns.A, Mns.B, Mns.C, Mns.D)
    Ad, Bd, Cd, Dd = balreal(AM, BM, CM, DM)
    # Initialise state, output and error
    xns = np.zeros((Ad.shape[0], 1))# noise-shaping filter state
    yns = np.zeros((1, 1))  # noise-shaping filter output
    e = np.zeros((1, 1))  # quantiser error

    C = np.zeros((1, X.size)).astype(int)  # save codes

    satcnt = 0  # saturation counter (generate warning if saturating)
    
    FB_ON = True  # turn on/off feedback, for testing
    
    for i in range(0, X.size):
        x = X[i]  # noise-shaper input
        # d = Dq[i]  # re-quantisation dither
        
        if FB_ON: w = x - yns[0, 0]  # use feedback
        else: w = x
        
        u = w 
        
        # Re-quantizer (mid-tread)
        q = math.floor(u/Qstep + 0.5)  # quantize
        c = q - math.floor(Vmin/Qstep)  # code

        # Saturation (can't index out of bounds)
        if c > 2**Nb - 1:
            c = 2**Nb - 1
            satcnt = satcnt + 1
            # if satcnt >= 10:
            #     print(f'warning: pos. sat. -- cnt: {satcnt}')
            
        if c < 0:
            c = 0
            satcnt = satcnt + 1
            # if satcnt >= 10:
            #     print(f'warning: neg. sat. -- cnt: {satcnt}')
        
        C[0, i] = c  # save code
        # Output models
        yi = YQns[c]  # ideal levels
        ym = MLns[c]  # measured levels
        
        # Generate error
        match QMODEL:  # model used in feedback
            case 1:  # ideal
                e[0] = yi - w
            case 2:  # measured/calibrated
                e[0] = ym - w
        
        # Noise-shaping filter
        xns = Ad@xns + Bd@e  # update state
        yns = Cd@xns  # update filter output

        # yns = signal.lfilter(bns, ans, e)
    return C

# NOise shaping
def noise_shaping(Nb, Xcs, b, a, Qstep, YQns, MLns, Vmin, QMODEL):

    YQns = YQns.squeeze()
    MLns = MLns.squeeze()

    C_NSD = np.zeros((1, Xcs.size)).astype(int)  # save codes
    # Error 
    err_buffer = np.zeros_like(b)

    u_ns = np.zeros_like(Xcs)

    for i in range(len(Xcs)):

        desired = Xcs[i] - np.dot(b, err_buffer)

        q = math.floor(desired/Qstep + 0.5)  # quantize

        c = q - math.floor(Vmin/Qstep)  # code

        if c > 2**Nb - 1:
            c = 2**Nb - 1
            
        if c < 0:
            c = 0

        C_NSD[0, i] = c  # save code

        match QMODEL:
            case 1:
                u_ns[i] = YQns[c]
            case 2:
                u_ns[i] = MLns[c]

        error = u_ns[i] - desired 

        err_buffer[0] = - error

        err_buffer[0] = - np.dot(a, err_buffer)

        err_buffer = np.roll(err_buffer, 1)

    return C_NSD

