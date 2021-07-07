# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:37:12 2021

@author: sebja
"""


import numpy as np 

from sklearn.isotonic import IsotonicRegression

from scipy import interpolate
from scipy import optimize

import matplotlib.pyplot as plt

class W_Stress:
    
    def __init__(self, y, G_P, u, gamma):
        # x, F, u, gamma
        
        self.ir = IsotonicRegression()
        self.y = y
        self.G_P = G_P(y)
        self.u = u
        
        #F_interp = interpolate.interp1d(x, F(x), kind='cubic')
        G_P_inv = lambda u : optimize.root_scalar( lambda y : (G_P(y) - u), method='bisect', bracket = [y[0],y[-1]])
        
        self.G_P_inv = np.zeros(len(u))
        for i in range(len(u)):
            self.G_P_inv[i] = G_P_inv(u[i]).root
            
        if isinstance(gamma, list):
            
            self.gamma = []
            for i in range(len(gamma)):
                this_gamma = gamma[i](self.u)
                this_gamma /= self.integrate(this_gamma, self.u)
                self.gamma.append(this_gamma)
            
        else:
            self.gamma = gamma(self.u)

    def ell_iso(self, lam):
        
        # eta = np.exp(lam)
        eta = 1.0* lam
        
        ell = 0 + self.G_P_inv 
        for i in range(len(self.gamma)):
            ell += eta[i]*self.gamma[i]
            
        
        return ell, self.ir.fit_transform(self.u, ell )

    def integrate(self, f, u):
        
        return np.sum(0.5*(f[:-1]+f[1:])*np.diff(u))

    def WassersteinDistance(self):
    
        return np.sqrt(self.integrate((self.Gs_inv-self.G_P_inv)**2, self.u))
    
    def RiskMeasure(self, G_inv):
        
        RM = np.zeros(len(self.gamma))
        for i in range(len(RM)):
            RM[i] = self.integrate( G_inv * self.gamma[i], self.u) 
            
        return RM
    
    def Plot_ell_iso(self, ell):
        
        
        fig = plt.figure(figsize=(4,4))
        plt.plot(self.u, ell, linewidth=0.5, color='g', label=r"$\ell$")
        plt.plot(self.u, self.Gs_inv, label = r"$G^{*-1}$", color='r')
        plt.plot(self.u, self.G_P_inv, linestyle='--', color='b', label=r"$F^{-1}_{Y}$")
        plt.legend(fontsize=14)
        
        plt.yscale('log')
        
#        plt.ylim( 0, 3 )
        
        plt.show()    
        
        return fig
    
    def Distribution(self, u, G_inv, x):
        
        print("using qtl derivative")
        
        dG_inv = (G_inv[2:]-G_inv[:-2])/(u[2:]-u[:-2])
        
        
        dG_inv_interp =  interpolate.interp1d( 0.5*(u[2:]+u[:-2]), dG_inv, kind='linear')
                
        eps = np.cumsum(1e-10*np.ones(len(G_inv)))
        G_interp =  interpolate.interp1d(eps+G_inv, u, kind='linear')
            
        G = G_interp(x)
        g = 1/dG_inv_interp(G_interp(x))
            
        return x, g, G
    
    
    def Optimise_RM(self, rm):
                                
        def Constraint_error(lam):
            
            ell, Gs_inv = self.ell_iso( lam )

            RM = self.RiskMeasure( Gs_inv )
            
            RM_Error = np.sqrt( np.sum( (rm - RM )**2) / len(self.gamma) )
                    
            return RM_Error
        

        KeepSearching = True
        
        lambda0 = np.zeros(len(rm))
        
        while KeepSearching:
            
            lambda0 = np.random.normal(size=(len(rm)))
            
            sol = optimize.minimize(Constraint_error, lambda0, method='Nelder-Mead',tol=1e-5)
            lam = sol.x
            
            ell, self.Gs_inv = self.ell_iso( lam )            
            
            KeepSearching = False
            
            if ( np.abs(rm- self.RiskMeasure(self.Gs_inv)) > 1e-4).any():
                KeepSearching = True
        
        print("lambda = ", lam)
        print(" WD = ", self.WassersteinDistance()  , end="\n")
        print(" Risk Measure = ", self.RiskMeasure(self.Gs_inv), end="\n")
        print(" Target Risk Measure = ", rm, end="\n")
        print(" Base Risk Measure = ", self.RiskMeasure(self.G_P_inv), end="\n")
        print("\n")      
        
        fig = self.Plot_ell_iso(ell)     
        
        return lam, self.WassersteinDistance(), self.RiskMeasure(self.Gs_inv), fig
    
    def MeanStd(self, G_inv):
        
        mean = self.integrate( G_inv, self.u)
        std = np.sqrt(self.integrate( (G_inv-mean)**2, self.u))
        
        return mean, std
    
    def ell_iso_MeanStd(self, lam, m):
        
        # eta = np.exp(lam)
        eta = 1.0* lam
        
        ell = (self.G_P_inv + eta[0] + eta[1] * m )/(1+eta[0])
        
        return ell, self.ir.fit_transform(self.u, ell )
    
    def Optimise_MeanStd(self, m, s):
        
        # self.iter = 0
                                
        def Constraint_error(lam):
            
            ell, gs = self.ell_iso_MeanStd( lam, m )

            mean, std = self.MeanStd( gs )
            
            error = 0.25*np.sqrt( (mean-m)**2 + (std-s)**2 )
                              
            return error
        

        KeepSearching = True
        
        while KeepSearching:
            
            lambda0 = np.random.normal(size=(2))
            
            sol = optimize.minimize(Constraint_error, lambda0, method='Nelder-Mead',tol=1e-5)
            lam = sol.x
            
            ell, self.Gs_inv = self.ell_iso_MeanStd( lam, m )            
            
            KeepSearching = False
            
            mean, std = self.MeanStd(self.Gs_inv)
            
            if ( np.abs(mean-m) > 1e-4):
                KeepSearching = True
            if ( np.abs(std-s) > 1e-4):
                KeepSearching = True
        
        mean_P, std_P = self.MeanStd(self.G_P_inv)
        
        print("lambda = ", lam)
        print(" WD = ", self.WassersteinDistance()  , end="\n")
        print(" Mean, Std = ", mean, std, end="\n")
        print(" Targets = ", m, s, end="\n")
        print(" Base Mean, Std = ", mean_P, std_P, end="\n")
        print("\n")      
        
        fig = self.Plot_ell_iso(ell)     
        
        return lam, self.WassersteinDistance(), [mean, std ], fig
    
        
    def UTransform(self, a, b, eta, u, G_inv, lam):
        
        g = np.zeros(len(u))
        
        nu = lambda x : x - lam * a * ( a/(1-eta)*x + b  )**(eta-1)
        
        for i in range(len(u)):
            g[i] = optimize.root_scalar( lambda x : nu(x) - G_inv[i], method='bisect', bracket = [-b*(1-eta)/a+1e-5,100]).root
            
        return g
    
    def Utility(self, a, b, eta, u, G_inv):
        
        return self.integrate( (1-eta)/eta * (a*G_inv/(1-eta) + b )**eta, u )
    
    def Optimise_HARA(self, a, b, eta, c, rm):
                  
        self.iter = 0     
         
        def Constraint_error(lam):

            
            
            ell, iso_g = self.ell_iso( lam[1:] )
            
            gs = self.UTransform(a, b, eta, self.u, iso_g, np.exp(lam[0]))
        

            RM = self.RiskMeasure( gs )
            Utility = self.Utility( a, b, eta, self.u, gs)
            
            self.iter+=1
            if np.mod(self.iter, 50)==0:
                print(lam, RM, Utility)                        
            
            
            error = np.sqrt( np.mean((RM-rm)**2) + (Utility- c)**2 )
                              
            return error
        

        KeepSearching = True
        
        while KeepSearching:
            
            lambda0 = np.random.normal(size=len(self.gamma)+1)
            
            sol = optimize.minimize(Constraint_error, lambda0, method='Nelder-Mead',tol=1e-5)
            lam = sol.x
            
            ell, iso_g = self.ell_iso( lam[1:] )
            
            self.Gs_inv = self.UTransform(a, b, eta, self.u, iso_g, np.exp(lam[0]) )

            RM = self.RiskMeasure( self.Gs_inv )
            Utility = self.Utility( a,b,eta, self.u, self.Gs_inv)            
            
            KeepSearching = False
            
            if ( np.abs(RM-rm) > 1e-4).any():
                KeepSearching = True
            if ( np.abs(Utility-c) > 1e-4):
                KeepSearching = True
        
        print("lambda = ", lam)
        print(" WD = ", self.WassersteinDistance()  , end="\n")
        print(" RM, Utility = ", RM, Utility, end="\n")
        print(" Targets = ", rm, c, end="\n")
        print(" Base = ", self.RiskMeasure(self.G_P_inv), self.Utility( a,b,eta, self.u, self.G_P_inv) , end="\n")
        print("\n")      
        
        fig = self.Plot_ell_iso(ell)     
        
        return lam, self.WassersteinDistance(), [RM, Utility], fig    