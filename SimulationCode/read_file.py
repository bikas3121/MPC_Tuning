
# %%
import scipy.io
nsf_num = scipy.io.loadmat('Optimal_NSF/NSF_num_100kHz_1MHz_3|2Mueta.mat')
nsf_den = scipy.io.loadmat('Optimal_NSF/NSF_den_100kHz_1MHz_3|2Mueta.mat')
bn = nsf_num['br']
an = nsf_den['ar']