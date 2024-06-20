
# %%
import numpy as np
import matplotlib.pyplot as plt
def plot_variance(**kwargs):
    # Plots the bar plots for the given values 
    x = []
    y = []
    for key, value in kwargs.items() :
        x.append(key)
        y.append(value)


    fig, ax = plt.subplots()
    x_pos = np.arange(0, len(x),1) 
    bar_labels = x
    ax.bar(x,y, label=bar_labels)
    plt.xticks([i  for i in range(len(y))],x[0:len(x)])
    for i in range(len(x)):
        plt.text(x = x_pos[i]-0.1, y= y[i] + 0.015*y[i] , s= y[i], size = 10)
    ax.set_ylabel('Error Variance')
    ax.legend()
    plt.show()

