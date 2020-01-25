import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as sp
import scipy.optimize as opt

def curvefit(model, data_x, data_y, data_sx = None, data_sy = None, xlabel = None, ylabel = None, latexModel = None):

    plt.figure(figsize=(10,10))

    # fit the data with the model (ans = array of the estimated parameters, cov = covariance matrix)
    if data_sy.any() != None:
        ans, cov = opt.curve_fit(model, data_x, data_y, sigma = data_sy)
    else:
        ans, cov = opt.curve_fit(model, data_x, data_y)
        
    fit_std = np.sqrt(np.diag(cov))

    # plot data & error bars
    if (data_sx.any() !=None and data_sy.any() != None):
        plt.errorbar(data_x, data_y, xerr = data_sx, yerr = data_sy, fmt = '.')
    elif(data_sx.any() != None):
        plt.errorbar(data_x, data_y, xerr = data_sx, fmt = '.')
    else:
        plt.errorbar(data_x, data_y, fmt = '.')
    if(data_sx.any() ==None and data_sy.any() != None):
        plt.errorbar(data_x, data_y, yerr = data_sy, fmt = '.')
        
    if ylabel != None:
        plt.ylabel(ylabel)
    if xlabel != None:
        plt.xlabel(xlabel)

    # plot curve fit
    labell = []
    llabel = ''
    latex = ''
    for i in range(len(ans)):
        labell.append(' $\\alpha_{} = {:.3f} \pm {:.3f}$\n'.format(int(i), ans[i], fit_std[i]))
        llabel += labell[i]
    if latexModel != None:
        latex = '$' + latexModel + '$'
    t = np.linspace(data_x.min(), data_x.max())
    plt.plot(t, model(t, *ans), label=llabel + latex)
    plt.legend()
