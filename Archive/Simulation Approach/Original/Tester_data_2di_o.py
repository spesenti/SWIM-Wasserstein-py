
# 2dimensional input variables X and an output Y

import copy
import numpy as np 
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

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
    
    
    y_P = np.linspace(StressModel.Gs_inv[5],StressModel.Gs_inv[-5],1000)
    y_Q = np.linspace(StressModel.Gs_inv[3],StressModel.Gs_inv[-3],1000)
    
    _, g_Q, G_Q = StressModel.Distribution(StressModel.u, StressModel.Gs_inv, y_Q)
    
    fig  = plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(y_Q, g_Q, color='r')
    plt.plot(y_P, g_P(y_P),'--', color='b')
    plt.ylim(bottom=0)
    plt.xlabel(r"$y$", fontsize=18)
    plt.ylabel(r"$g^{*}(y)$", fontsize=18)
    plt.xlim(1, 8)
    
    plt.subplot(1,2,2)
    plt.plot(y_Q, G_Q, color='r')
    plt.plot(y_P, G_P(y_P),'--', color='b')
    plt.xlabel(r"$y$", fontsize=18)
    plt.ylabel(r"$G^{*}(y)$", fontsize=18)
    plt.xlim(1, 8)
    plt.ylim(0,1.05)
    
    plt.tight_layout()
    
    plt.show()
    
    fig.savefig(filename + '.pdf',format='pdf')
       
    idx = np.where( np.diff(StressModel.Gs_inv)<1e-8)[0][0]
    
    fig = plt.figure(figsize=(4,4))
    
    dQ_dP = g_Q/ g_P(y_Q)
    if type == "ES":
        dQ_dP[:idx] = 1
    plt.plot(y_Q, dQ_dP)
        
    plt.ylim(0,10)
    plt.xlim(3.8,10)
    plt.xlabel(r"$y$", fontsize=18)
    plt.ylabel(r"$\frac{g^{*}(y)}{f(y)}$", fontsize=22)
    plt.tight_layout()
    fig.savefig(filename + '_RN.pdf',format='pdf')
    plt.show()


    idx = (data["y"]>= y_Q[0]) & (data["y"] <= y_Q[-1])
    data_y_idx = data["y"][idx]    

    w = StressModel.gs(data_y_idx) / StressModel.f(data_y_idx)
    
    data_x_idx = data["x"][idx]
    
    f_x_plt = f_x(x1, x2, data_x_idx)
    gs_x_plt = gs_x(x1, x2, w,  data_x_idx)
    
    plt.figure(figsize=(8,4))
    
    plt.subplot(1,2,1)
    plt.contour(x1, x2, f_x_plt)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18)
    
    plt.subplot(1,2,2)
    plt.contour(x1, x2, gs_x_plt)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18)
    
    plt.tight_layout()
    plt.show()
    
    fig = plt.figure(figsize=(4,4))
    f_x_plt_ctr = plt.contour(x1, x2, f_x_plt, alpha=0.8)
    plt.scatter(data_x_idx[:,0], data_x_idx[:,1], color='r',s=0.1)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18)
    plt.xlim(-1, 3)
    plt.ylim(-1, 3)
    plt.tight_layout()
    fig.savefig(filename + '_data_x_P.pdf',format='pdf')
    plt.show()
    
    fig = plt.figure(figsize=(4,4))
    f_x_plt_ctr = plt.contour(x1, x2, f_x_plt, alpha=0.4)
    gs_x_plt_ctr = plt.contour(x1, x2, gs_x_plt, levels = f_x_plt_ctr.levels )
    plt.scatter(data_x_idx[:,0], data_x_idx[:,1], color='r',s=0.1)
    plt.xlim(-1, 3)
    plt.ylim(-1, 3)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18)
    plt.tight_layout()
    fig.savefig(filename + '_data_x_Q.pdf',format='pdf')
    plt.show()

#%% Basic Metrics
def Metrics(data, w):
    
    mean_y_P = np.mean(data["y"],axis=0)
    std_y_P = np.std(data["y"],axis=0)
    
    mean_x_P = np.mean(data["x"],axis=0)
    cov_x_P = np.cov(data["x"].T)
    corr_x_P = cov_x_P[0,1]/np.sqrt(cov_x_P[0,0]*cov_x_P[1,1])
    
    P ={"mean_x" : mean_x_P, "cov_x" : cov_x_P, "corr_x" : corr_x_P, "mean_y" : mean_y_P, "std_y" : std_y_P  }


    mean_y_Q = np.sum(data["y"] * w ,axis=0) / np.sum(w)
    std_y_Q = np.sqrt(np.sum((data["y"]-mean_y_Q)**2 * w ,axis=0)  / np.sum(w))

    
    mean_x_Q = np.sum(data["x"] * np.matlib.repmat(w.reshape(-1,1),1,2),axis=0)/np.sum(w)
    cov_x_Q = np.zeros((2,2))
    cov_x_Q[0,0] = np.sum((data["x"][:,0]-mean_x_Q[0])**2 * w,axis=0)/np.sum(w)
    cov_x_Q[0,1] = np.sum((data["x"][:,0]-mean_x_Q[0])*(data["x"][:,1]-mean_x_Q[1]) * w,axis=0)/np.sum(w)
    cov_x_Q[1,1] = np.sum((data["x"][:,1]-mean_x_Q[1])**2 * w,axis=0)/np.sum(w)
    cov_x_Q[1,0] =  cov_x_Q[0,1]
    
    corr_x_Q = cov_x_Q[0,1]/np.sqrt(cov_x_Q[0,0]*cov_x_Q[1,1])   
    
    Q ={"mean_x" : mean_x_Q, "cov_x" : cov_x_Q, "corr_x" : corr_x_Q, "mean_y" : mean_y_Q, "std_y" : std_y_Q  }
    
    
    print("P", P)
    print("Q",Q)
    print("mean_x", Q["mean_x"]/P["mean_x"]-1)
    print("std_x", np.sqrt(np.diag(Q["cov_x"])/np.diag(P["cov_x"]))-1)
    print("corr_x", Q["corr_x"]/P["corr_x"]-1)
    print("std_y", Q["std_y"]/P["std_y"]-1)
    
    return P, Q

#%
def GenerateWeights(data, StressModel):
    
    
    y_gd = np.linspace(StressModel.Gs_inv[3],StressModel.Gs_inv[-3],500)
    
    print("computing gs at grid points...")
    gs = StressModel.gs(y_gd)
    gs /= integrate(gs, y_gd)
    
    print("computing f at grid points...")
    f = StressModel.f(y_gd)
    f /= integrate(f, y_gd)
    
    dQ_dP = gs/f
    
    print("E[dQ/dP]", integrate(dQ_dP * f, y_gd))
    
    print("computing weights...")
    w = np.zeros(len(data["y"]))
    for i in range(len(data["y"])):
        
        w[i] = integrate( norm.pdf( (y_gd - data["y"][i]) / h_y)/h_y *dQ_dP, y_gd )
        # print( i, " of ", len(data["y"]))
        
    return w

#%% mean and ES_0.95
alpha = [0, 0.95]

y = np.linspace(1e-20, 30, 1000)
u = Create_u_grid([0.95])


gamma = [lambda u : (u>alpha[0])/(1-alpha[0]), lambda u : (u>alpha[1])/(1-alpha[1])]

for i in range(len(gamma)):
    plt.plot(u, gamma[i](u))
plt.ylabel(r'$\gamma(u)$')
plt.xlabel('u')
plt.show()


#%% generate some fake data
Nsims = 1000
p_mix = 0.5
H = ( np.random.uniform(size=Nsims) < p_mix)

params = {"H1" : {"mu" : np.array([0.5,1]), "cov" : np.array([[0.2,0.1],[0.1,0.4]])}, 
          "H2" : {"mu" : np.array([1,0.5]), "cov" : np.array([[0.6,-0.3],[-0.3,0.3]])} }


lognormal = lambda x, mu, cov : np.exp( np.matlib.repmat( (mu - 0.5*np.diag(cov) ).reshape(1,-1), Nsims, 1) + x )

n1 = np.random.multivariate_normal(mean=params["H1"]["mu"], cov=params["H1"]["cov"], size = Nsims)
l1 = lognormal(n1, params["H1"]["mu"], params["H1"]["cov"])

n2 = np.random.multivariate_normal(mean=params["H2"]["mu"], cov=params["H2"]["cov"], size = Nsims)
l2 = lognormal(n2, params["H2"]["mu"], params["H2"]["cov"])

x=np.matlib.repmat((H==0).reshape(-1,1), 1,2)* n1 + np.matlib.repmat((H==1).reshape(-1,1), 1,2) * n2

g = sns.jointplot(x=x[:,0],y=x[:,1], s=2)
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
plt.xlim(-1,4)
plt.ylim(-1,4)
plt.xlabel(r"$x_1$", fontsize=18)
plt.ylabel(r"$x_2$", fontsize=18)
plt.tight_layout()
plt.savefig("data_x.pdf", type="pdf")

plt.show()

data = {"y": 2+np.exp(x[:,0]+x[:,1]-3).reshape(-1), "x" : x}

h_y = 1.06*np.std(data["y"])*(len(data["y"]))**(-1/5)/2
f = lambda y : np.sum( norm.pdf( (y.reshape(-1,1)-data["y"].reshape(1,-1))/h_y )/h_y /len(data["y"]), axis=1).reshape(-1)

F = lambda y : np.sum( norm.cdf( (y.reshape(-1,1)-data["y"].reshape(1,-1))/h_y ) /len(data["y"]), axis=1).reshape(-1)        

# bivariate KDE for x1, x2

phi = lambda z, z_data, h : norm.pdf( (z.reshape(-1,1)-z_data.reshape(1,-1))/h )/h

x1, x2 = np.meshgrid( np.linspace(-1,4, 100), np.linspace(-1,4, 100))

h_x = 1.06*np.std(data["x"],axis=0)*(len(data["y"]))**(-1/5)

def f_x(x1, x2, x_data):
    
    f = np.zeros(x1.shape)
    
    for i in range(len(x_data)):
        
        f+= norm.pdf(x1, x_data[i,0], h_x[0]) *norm.pdf(x2, x_data[i,1], h_x[1])
       
    f /= len(x_data)
    
    return f

plt.figure(figsize=(4,4))
plt.contour(x1, x2, f_x(x1, x2, data["x"]))
plt.xlabel(r"$x_1$", fontsize=18)
plt.ylabel(r"$x_2$", fontsize=18)
plt.show()

def gs_x(x1, x2, w, x_data):
    
    gs = np.zeros(x1.shape)
    
    for i in range(len(x_data)):
        
        gs += w[i]*norm.pdf(x1, x_data[i,0], h_x[0]) *norm.pdf(x2, x_data[i,1], h_x[1])
        
    gs /= np.sum(w)
    
    return gs

#%% Generate the model
StressModel = W_Stress( data, u, gamma)

RM_P = StressModel.RiskMeasure(StressModel.F_inv)
#%% Optimis RM
lam, WD, RM_Q, fig = StressModel.Optimise_RM(RM_P*np.array([1.05,1.05]))

filename = 'data_ES_0_u5_95_u5_v2'

fig.savefig(filename + '_inv.pdf',format='pdf')

PlotDist(StressModel, filename, f, F, "ES")

w =  GenerateWeights(data, StressModel)

P, Q = Metrics(data, w)


#%% Optimis RM
lam, WD, RM_Q, fig = StressModel.Optimise_RM(RM_P*np.array([0.95,1.05]))

filename = 'data_ES_0_d5_95_u5p_v2'

fig.savefig(filename + '_inv.pdf',format='pdf')

PlotDist(StressModel, filename, f, F, "ES")

w =  GenerateWeights(data, StressModel)

P, Q = Metrics(data, w)

#%% Test Mean and Variance Optimisation
mean_P, std_P = StressModel.MeanStd(StressModel.F_inv)
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


