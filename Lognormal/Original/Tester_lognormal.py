import copy
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm


from W_Stress import W_Stress

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
    x_Q = np.linspace(StressModel.Gs_inv[2],10,1000)
    _, f_Q, F_Q = StressModel.Distribution(StressModel.u, StressModel.Gs_inv, x_Q)
    
    fig  = plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(x_Q, f_Q, color='r')
    plt.plot(y_P, g_P(y_P),'--', color='b')
    plt.ylim(0,0.5)
    plt.xlabel(r"$y$", fontsize=18)
    plt.ylabel(r"$g^{*}(y)$", fontsize=18)
    plt.xlim(0, 10)
    
    plt.subplot(1,2,2)
    plt.plot(x_Q, F_Q, color='r')
    plt.plot(y_P, G_P(y_P),'--', color='b')
    plt.xlabel(r"$y$", fontsize=18)
    plt.ylabel(r"$G^{*}(y)$", fontsize=18)
    plt.xlim(0, 10)
    plt.ylim(0,1.05)
    
    plt.tight_layout()
    
    plt.show()
    
    # fig.savefig(filename + '.pdf',format='pdf')
     
    mask = ( np.diff(StressModel.Gs_inv)<1e-10 )
    if np.sum(mask) > 0:
        idx = np.where(mask )[0][0]
    else:
        idx = []
    
    fig = plt.figure(figsize=(4,4))
    
    dQ_dP = f_Q/ g_P(x_Q)
    if type == "ES":
        dQ_dP[:idx] = 1
    plt.plot(x_Q, dQ_dP)
        
    plt.ylim(0,2)
    plt.xlabel(r"$y$", fontsize=18)
    plt.ylabel(r"$\frac{g^{*}(y)}{f(y)}$", fontsize=22)
    plt.tight_layout()
    # fig.savefig(filename + '_RN.pdf',format='pdf')
    plt.show()


#%% ES_0.8 and ES_0.95 with log-normal(mu, sigma)

# log-normal params
mu = 1
sigma = 0.5

alpha = [0.8, 0.95] # is the collection of ES_alpha levels

# grids in random variable space (0,\infty), and in quantile space \in (0,1)
y = np.linspace(1e-20, 30, 1000)
u = Create_u_grid(alpha)

# this is a collection of lambda's that encode the constraints
gamma = [lambda u : (u>alpha[0])/(1-alpha[0]), lambda u : (u>alpha[1])/(1-alpha[1])]

for i in range(len(gamma)):
    plt.plot(u, gamma[i](u))
plt.ylabel(r'$\gamma(u)$')
plt.xlabel('u')
plt.show()

# log-normal  pdf and cdf
g_P = lambda y : norm.pdf( (np.log(y)-(mu-0.5*sigma**2))/sigma)/(y*sigma)
G_P = lambda y : norm.cdf( (np.log(y)-(mu-0.5*sigma**2))/sigma)

#%% Generate the model
StressModel = W_Stress( y, G_P, u, gamma)

plt.plot(u,StressModel.G_P_inv)
plt.plot(G_P(y),y)
plt.yscale('log')
plt.ylim(1e-3,1e2)
plt.show()

# compute the risk-measure of the based model
RM_P = StressModel.RiskMeasure(StressModel.G_P_inv)

# #%% Optimis RM
# lam, WD, RM_Q, fig = StressModel.Optimise_RM(RM_P*np.array([1.1,1.1]))
#
# filename = 'lognormal_ES_80_95_s10'
#
# # fig.savefig(filename + '_inv.pdf',format='pdf')
#
# PlotDist(StressModel, filename, g_P, G_P, "ES")
#
#
# #%% Test Mean and Variance Optimisation
# mean_P, std_P = StressModel.MeanStd(StressModel.Gs_inv)
#
# lam, WD, mv_Q, fig = StressModel.Optimise_MeanStd(mean_P, 1.2*std_P)
#
# filename = 'lognormal_utility_MS_20'
# # fig.savefig(filename + '_inv.pdf',format='pdf')
#
# PlotDist(StressModel, filename, g_P, G_P, "mean-std")


#%% Test Utility and risk measure
hara = lambda a, b, eta, x : (1-eta)/eta * (a*x/(1-eta) + b )**eta

b = lambda eta : 5*(eta/(1-eta))**(1/eta)
x = np.linspace(0.001,2,100)
plt.plot(x,hara(1, b(0.2), 0.2, x))

RM_P = StressModel.RiskMeasure(StressModel.G_P_inv)

Utility_P = StressModel.Utility(1, b(0.2), 0.2, StressModel.u, StressModel.G_P_inv)

_, _, _, fig  = StressModel.Optimise_HARA(1, b(0.2), 0.2, Utility_P*1.01, RM_P*np.array([0.9,1.1]))

filename = 'lognormal_utility_rm_3_90_110'
# fig.savefig(filename + '_inv.pdf',format='pdf')

PlotDist(StressModel, filename, g_P, G_P, "Utility")

