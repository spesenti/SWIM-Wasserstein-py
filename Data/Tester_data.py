# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:51:39 2021

@author: sebja
"""


import copy
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm

from W_Stress_data import W_Stress

# %% Helper Functions
def integrate(f, x):
    
    return np.sum(0.5*(f[:-1]+f[1:])*np.diff(x))


def Create_a_b_grid(a, b, N):
    
    eps=0.002    
    
    u_eps = 10**(np.linspace(-10, np.log(eps)/np.log(10),10))-1e-11
    u_eps_flip = np.flip(copy.deepcopy(u_eps)) 

    u1 = a + u_eps
    u2 = np.linspace(a + eps, b - eps, N)
    u3 = b - u_eps_flip
    
    return np.concatenate((u1,u2, u3))


def Create_u_grid(pts):
     
    eps = 1e-5

    knots = np.sort(pts)

    u = Create_a_b_grid(eps, knots[0], 100)
    for i in range(1,len(knots)):
        u = np.concatenate((u, Create_a_b_grid(knots[i-1], knots[i], 500)))

    u = np.concatenate((u,Create_a_b_grid(knots[-1], 1-eps, 100) ))
    
    return u

def PlotDist(StressModel, filename, g_P, G_P, type =""):
    
    
    y_P = np.linspace(0.01,10,500)
    x_Q = np.linspace(StressModel.Gs_inv[3],StressModel.Gs_inv[-3],1000)
    _, g_Q, G_Q = StressModel.Distribution(StressModel.u, StressModel.Gs_inv, x_Q)
    
    fig  = plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(x_Q, g_Q, color='r')
    plt.plot(y_P, g_P(y_P),'--', color='b')
    plt.ylim(bottom=0)
    plt.xlabel(r"$y$", fontsize=18)
    plt.ylabel(r"$g^{*}(y)$", fontsize=18)
    plt.xlim(3.8, 10)
    
    plt.subplot(1,2,2)
    plt.plot(x_Q, G_Q, color='r')
    plt.plot(y_P, G_P(y_P),'--', color='b')
    plt.xlabel(r"$y$", fontsize=18)
    plt.ylabel(r"$G^{*}(y)$", fontsize=18)
    plt.xlim(3.8, 10)
    plt.ylim(0,1.05)
    
    plt.tight_layout()
    
    plt.show()
    
    fig.savefig(filename + '.pdf',format='pdf')
       
    idx = np.where( np.diff(StressModel.Gs_inv)<1e-8)[0][0]
    
    fig = plt.figure(figsize=(4,4))
    
    dQ_dP = g_Q/ g_P(x_Q)
    if type == "ES":
        dQ_dP[:idx] = 1
    plt.plot(x_Q, dQ_dP)
        
    plt.ylim(0,10)
    plt.xlim(3.8,10)
    plt.xlabel(r"$y$", fontsize=18)
    plt.ylabel(r"$\frac{g^{*}(y)}{f(y)}$", fontsize=22)
    plt.tight_layout()
    fig.savefig(filename + '_RN.pdf',format='pdf')
    plt.show()


#%% ES_0.9 and ES_0.95 with log-normal
mu = 1
sigma = 0.5

alpha = [0, 0.95]

y = np.linspace(1e-20, 30, 1000)
u = Create_u_grid([0.95])


gamma = [lambda u : (u>alpha[0])/(1-alpha[0]), lambda u : (u>alpha[1])/(1-alpha[1])]

# gamma = [lambda u : (u>alpha[0])/(1-alpha[0])]

for i in range(len(gamma)):
    plt.plot(u, gamma[i](u))
plt.ylabel(r'$\gamma(u)$')
plt.xlabel('u')
plt.show()


#%% generate some fake data
Nsims = 100
p_mix = 0.2
H = ( np.random.uniform(size=Nsims) < p_mix)
data = {"y": np.random.normal(loc=7*H+(1-H)*5, scale=(1+H)/5)}

h_y = 1.06*np.std(data["y"])*(len(data["y"]))**(-1/5)/2
f = lambda y : np.sum( norm.pdf( (y.reshape(-1,1)-data["y"].reshape(1,-1))/h_y )/h_y /len(data["y"]), axis=1).reshape(-1)

F = lambda y : np.sum( norm.cdf( (y.reshape(-1,1)-data["y"].reshape(1,-1))/h_y ) /len(data["y"]), axis=1).reshape(-1)        

#%% Generate the model
StressModel = W_Stress( data, u, gamma)

RM_P = StressModel.RiskMeasure(StressModel.F_inv)
#%% Optimis RM
lam, WD, RM_Q, fig = StressModel.Optimise_RM(RM_P*np.array([1.05,1.05]))

filename = 'data_ES_0_u5_95_u5'

fig.savefig(filename + '_inv.pdf',format='pdf')

PlotDist(StressModel, filename, f, F, "ES")


#%% Optimis RM
lam, WD, RM_Q, fig = StressModel.Optimise_RM(RM_P*np.array([0.95,1.05]))

filename = 'data_ES_0_d5_95_u5p'

fig.savefig(filename + '_inv.pdf',format='pdf')

PlotDist(StressModel, filename, f, F, "ES")

#%% Test Mean and Variance Optimisation
mean_P, std_P = StressModel.MeanStd(StressModel.G_P_inv)
lam, WD, mv_Q, fig = StressModel.Optimise_MeanStd(mean_P, 1.2*std_P)

filename = 'data_MS_20'
fig.savefig(filename + '_inv.pdf',format='pdf')

PlotDist(StressModel, filename, f, F)


#%% Test Utility ******** NOT Converging ********
hara = lambda a, b, eta, x : (1-eta)/eta * (a*x/(1-eta) + b )**eta

b = lambda eta : 5*(eta/(1-eta))**(1/eta)
plt.plot(y, hara(1, b(0.2), 0.2, y))

RM_P = StressModel.RiskMeasure(StressModel.F_inv)
Utility_P = StressModel.Utility(1, b(0.2), 0.2, StressModel.u, StressModel.F_inv)

_, _, _, fig  = StressModel.Optimise_HARA(1, b(0.2), 0.2, Utility_P*1, RM_P*np.array([1.0,1.0]))

filename = 'data_utility_rm_1_0_95_s5_downup'
fig.savefig(filename + '_inv.pdf',format='pdf')

PlotDist(StressModel, filename, f, F)
