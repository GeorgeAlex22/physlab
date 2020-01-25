import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def curvefit(model, data_x, data_y, data_sx = None, data_sy = None, xlabel = None, ylabel = None, latexModel = None, size = (10,10), savename = None, title = None):
    """

    :param model: The function which you want the data to fit
    :param data_x: The data along x-axis (ndarray)
    :param data_y: The data along y-axis (ndarray)
    :param data_sx: The standard deviations of the data along x-axis (ndarray)
    :param data_sy: The standard deviations of the data along y-axis (ndarray)
    :param xlabel: The x-axis name (str)
    :param ylabel: The y-axis name (str)
    :param latexModel: String in LaTeX code w/o the '$' to show the model function (name the parameters with \\alpha_i , i: number of parameters count from 0)
    :param size: Tuple for the plt.figure( figsize = )
    :param savename: String with the file name (no extension) saves as .png
    :param title: Title of the plot (str)
    :return: Returns the list: [Parameters, Covariance Matrix]
    """
    plt.figure(figsize = size)

    # fit the data with the model (ans = array of the estimated parameters, cov = covariance matrix)
    if data_sy.any() is not None and data_sy.any() != 0:
        ans, cov = opt.curve_fit(model, data_x, data_y, sigma = data_sy)
    else:
        ans, cov = opt.curve_fit(model, data_x, data_y)
        
    fit_std = np.sqrt(np.diag(cov))

    # plot data & error bars

    if data_sx.any() is not None and data_sy.any() is None:
        data_sy = np.zeros(len(data_sx))

    if data_sy.any() is None and data_sx.any() is None:
        data_sx = np.zeros(len(data_sx))
        data_sy = np.zeros(len(data_sx))

    if data_sx.any() is None and data_sy.any() is not None:
        data_sx = np.zeros(len(data_sx))

    plt.errorbar(data_x, data_y, xerr=data_sx, yerr=data_sy, fmt='k.', capsize = 3, label = 'data', ecolor = 'k',)

    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlabel is not None:
        plt.xlabel(xlabel)

    # plot curve fit
    labell = []
    llabel = ''
    latex = ''
    for i in range(len(ans)):
        labell.append(' $\\alpha_{} = {:.3f} \pm {:.3f}$\n'.format(int(i), ans[i], fit_std[i]))
        llabel += labell[i]
    if latexModel is not None:
        latex = 'fitting model: $' + latexModel + '$'
    t = np.linspace(data_x.min(), data_x.max())
    plt.plot(t, model(t, *ans), label=llabel + latex)
    plt.legend()
    if savename is not None:
        plt.savefig(savename+'.png')

    if title is not None:
        plt.title(title)
    plt.show()

    return [ans, cov]
