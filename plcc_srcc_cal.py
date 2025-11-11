import numpy as np
import torch
import os
import argparse
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt




def logistic(X, beta1, beta2, beta3, beta4, beta5):
    logistic_part = 0.5 - 1./(1 + np.exp(beta2 * (X - beta3)))
    yhat = beta1 * logistic_part + beta4 * X + beta5
    return yhat

def correlation_evaluation(obj_score, mos,is_plot, plot_path, xlabel='Fitted score', ylabel='MOS'):
    beta1 = np.max(mos)
    beta2 = np.min(mos)
    beta3 = np.mean(obj_score)
    beta = [beta1, beta2, beta3, 0.1, 0.1]  # inital guess for non-linear fitting

    obj_score = np.array(obj_score)
    mos = np.array(mos)
    try:
        popt, _ = curve_fit(logistic, xdata=obj_score, ydata=mos, p0=beta, maxfev=10000)
    except:
        popt = beta

    ypred = logistic(obj_score, popt[0], popt[1], popt[2], popt[3], popt[4])

    plcc, _ = pearsonr(mos, ypred)
    srcc, _ = spearmanr(mos, ypred)
    rmse = np.sqrt(np.mean((ypred - mos) ** 2))
    
    if is_plot:
        obj_score = ypred
        myfonts = "Times New Roman"
        matplotlib.rcParams['font.sans-serif'] = myfonts
        # print('obj_score:',obj_score, 'min:', min(obj_score), 'max:', max(obj_score))
        # _pseu_x = np.linspace(np.min(obj_score), np.max(obj_score), 100)
        # _pseu_pred = logistic(_pseu_x, popt[0], popt[1], popt[2], popt[3], popt[4])
        plt.style.use('ggplot')

        fig = plt.figure()
        plt.plot(obj_score, mos, marker="o", color='royalblue',markersize=4, linestyle='')
        plt.title('PLCC: {:0.3f}, SRCC: {:0.3f}'.format( plcc, srcc),fontsize=20)
        plt.xlabel(xlabel, color='black',fontsize=18)
        plt.ylabel(ylabel, color='black',fontsize=18)
        plt.xticks(color='black',fontsize=18)
        plt.yticks(color='black',fontsize=18)
        fig.set_figheight(6)
        fig.savefig(plot_path, dpi=400)
        plt.close()
        
    return float(plcc), float(srcc), float(rmse)

    
# if __name__ == '__main__':

#     corr = {}
#     mos = np.reshape(np.asarray(self.mos), (-1,))
#     pred = np.reshape(np.asarray(self.pred), (-1,))
#     plcc, srcc, rmse = correlation_evaluation(pred, mos,is_plot=self.isplot, plot_path=self.plot_path)
#     corr['plcc'], corr['srcc'], corr['rmse'] = np.round((plcc, srcc, rmse), 4)
    
    
    
    