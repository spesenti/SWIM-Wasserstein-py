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

    def __init__(self, y, G_P, u):
        # x, F, u, gamma

        self.ir = IsotonicRegression()
        self.y = y
        self.G_P = G_P(y)
        self.u = u

        # Inverse G_P
        G_P_inv = lambda u: optimize.root_scalar(lambda y: (G_P(y) - u), method='bisect', bracket=[y[0], y[-1]])

        self.G_P_inv = np.zeros(len(u))
        for i in range(len(u)):
            self.G_P_inv[i] = G_P_inv(u[i]).root

        self.G_Q_inv = None

        # Initialize gammas (for risk constraints)
        self.gammas = []

    # Static functions
    @staticmethod
    def integrate(f, u):
        # integral of f over space x using Reimann sums
        return np.sum(0.5 * (f[:-1] + f[1:]) * np.diff(u))

    @staticmethod
    def distribution(u, G_inv, x):

        print("using qtl derivative")

        dG_inv = (G_inv[2:] - G_inv[:-2]) / (u[2:] - u[:-2])

        dG_inv_interp = interpolate.interp1d(0.5 * (u[2:] + u[:-2]), dG_inv, kind='linear')

        eps = np.cumsum(1e-10 * np.ones(len(G_inv)))
        G_interp = interpolate.interp1d(eps + G_inv, u, kind='linear')

        G = G_interp(x)
        g = 1 / dG_inv_interp(G_interp(x))

        return x, g, G

    # Plot functions
    def plot_ell_iso(self, ell, title=""):

        fig = plt.figure(figsize=(4, 4))
        plt.plot(self.u, ell, linewidth=0.5, color='g', label=r"$\ell$")
        plt.plot(self.u, self.G_Q_inv, label=r"$\breve{G}^*_Y$", color='r')
        plt.plot(self.u, self.G_P_inv, linestyle='--', color='b', label=r"$\breve{F}_Y$")
        plt.title(title)
        plt.legend(fontsize=14)

        plt.yscale('log')

        plt.show()

        return fig

    def plot_ell(self, ell):

        fig = plt.figure(figsize=(4, 4))
        plt.subplot(2, 1, 1)
        plt.plot(self.u, ell, linewidth=0.5, color='g', label=r"$\ell$")
        plt.legend(fontsize=14)
        plt.yscale('log')

        plt.subplot(2, 1, 2)
        plt.plot(self.u, ell, linewidth=0.5, color='g', label=r"$\ell$")
        plt.plot(self.u, self.G_Q_inv, label=r"$\breve{G}^*_Y$", color='r')
        plt.legend(fontsize=14)
        plt.yscale('log')

        plt.show()

        return fig

    # Get statistics (Wasserstein distance, risk measure, mean, standard deviation)
    def wasserstein_distance(self):
        # Calculate the Wasserstein distance W2(G, F)
        return np.sqrt(self.integrate((self.G_Q_inv - self.G_P_inv) ** 2, self.u))

    def get_risk_measure(self, G_inv):
        # Calculate risk measure rho = int {G_inv * gamma} du for each gamma
        if not self.gammas:
            print("Please set gamma functions before calling get_risk_measure.")
            return

        RM = np.zeros(len(self.gammas))
        for i in range(len(RM)):
            RM[i] = self.integrate(G_inv * self.gammas[i], self.u)

        return RM

    def get_risk_measure_baseline(self):
        return self.get_risk_measure(self.G_P_inv)

    def get_risk_measure_stressed(self):
        if self.G_Q_inv is None:
            print("Stressed distribution does not exist.")
            return None
        else:
            return self.get_risk_measure(self.G_Q_inv)

    def get_mean_std(self, G_inv):

        # Calculate mean of distribution E[X] = int{1 - G(x) dx} = int{G_inv(u) du}
        mean = self.integrate(G_inv, self.u)

        # Calculate standard deviation of distribution
        std = np.sqrt(self.integrate((G_inv - mean) ** 2, self.u))

        return mean, std

    def get_mean_std_baseline(self):
        return self.get_mean_std(self.G_P_inv)

    def get_mean_std_stressed(self):
        if self.G_Q_inv is None:
            print("Stressed distribution does not exist.")
            return None
        else:
            return self.get_mean_std(self.G_Q_inv)

    def get_hara_utility(self, a, b, eta, u, G_inv):
        return self.integrate((1 - eta) / eta * (a * G_inv / (1 - eta) + b) ** eta, u)

    def UTransform(self, a, b, eta, u, G_inv, lam):

        g = np.zeros(len(u))

        # nu(x) = x - lam * u'(x)
        # u(x) = (1 - eta)/eta * (a * x / (1-eta) + b)^eta
        # u'(x) = a * (a * x / (1-eta) + b)^(eta - 1)
        nu = lambda x: x - lam * a * (a / (1 - eta) * x + b) ** (eta - 1)

        # Get g = nu_inv(.)
        for i in range(len(u)):
            g[i] = optimize.root_scalar(lambda x: nu(x) - G_inv[i], method='bisect',
                                        bracket=[-b * (1 - eta) / a + 1e-5, 100]).root

        return g

    # gamma functions
    def set_gamma(self, gammas):
        self.gammas = []  # Reset gammas
        for gamma in gammas:
            this_gamma = gamma(self.u)
            this_gamma /= self.integrate(this_gamma, self.u)
            self.gammas.append(this_gamma)
        return

    def add_gamma(self, gammas):
        for gamma in gammas:
            this_gamma = gamma(self.u)
            this_gamma /= self.integrate(this_gamma, self.u)
            self.gammas.append(this_gamma)
        return

    # ell functions
    def ell_rm(self, lam):
        # ell = F_inv(u) + sum{lambda * gamma(u)}
        ell = self.G_P_inv.copy()

        for i in range(len(self.gammas)):
            ell += lam[i] * self.gammas[i]

        return ell

    def ell_mean_std(self, lam, m):
        # ell = 1/(1 + lambda2) * (F_inv(u) + lambda_1 + lambda_2 * m)
        ell = (self.G_P_inv + lam[0] + lam[1] * m) / (1 + lam[1])

        return ell

    def ell_rm_mean_std(self, lam, m):
        # ell = 1/(1 + lambda2) * (F_inv(u) + lambda_1 + lambda_2 * m + sum{lambda_{k+2} + gamma_k(u)})

        ell_partial = self.ell_rm(lam[2:])  # ell_partial = F_inv(u) + sum{lambda_{k+2} * gamma_k(u)}
        ell = (lam[0] + lam[1] * m + ell_partial) / (1 + lam[1])

        return ell

    def get_iso(self, ell):
        return self.ir.fit_transform(self.u, ell)

    # optimise functions
    def optimise_rm(self, rm, title=""):
        # Solve optimization problem with risk measure constraints gammas
        # min. W2(G,F) s.t. rho_gamma(G) = r for each risk measure

        if not self.gammas:
            print("Please set gamma functions before calling optimise.")
            return

        # Calculate the error for set of Lagrange multipliers lambda
        def constraint_error(lam):
            # Get the stressed inverse distribution G_Q_inv
            ell = self.ell_rm(lam)
            G_Q_inv = self.get_iso(ell)

            # Get stressed risk measure RM and calculate the error to minimize
            RM = self.get_risk_measure(G_Q_inv)
            RM_Error = np.sqrt(np.sum((rm - RM) ** 2) / len(self.gammas))

            return RM_Error

        search = True

        while search:

            lambda0 = np.random.normal(size=(len(rm)))

            sol = optimize.minimize(constraint_error, lambda0, method='Nelder-Mead', tol=1e-5)
            lam = sol.x

            ell = self.ell_rm(lam)
            self.G_Q_inv = self.get_iso(ell)

            if not (np.abs(rm - self.get_risk_measure_stressed()) > 1e-4).any():
                search = False

        print("lambda = ", lam)
        print(" WD = ", self.wasserstein_distance(), end="\n")
        print(" Risk Measure = ", self.get_risk_measure_stressed(), end="\n")
        print(" Target Risk Measure = ", rm, end="\n")
        print(" Base Risk Measure = ", self.get_risk_measure_baseline(), end="\n")
        print("\n")

        fig = self.plot_ell_iso(ell, title)

        return lam, self.wasserstein_distance(), self.get_risk_measure_stressed(), fig

    def optimise_mean_std(self, m, s, title=""):

        def constraint_error(lam):

            ell = self.ell_mean_std(lam, m)
            G_Q_inv = self.get_iso(ell)
            mean, std = self.get_mean_std(G_Q_inv)
            error = np.sqrt(2) * np.sqrt((mean - m) ** 2 + (std - s) ** 2)  # sqrt(2) normalization constant

            return error

        search = True

        while search:

            lambda0 = np.random.normal(size=2)

            sol = optimize.minimize(constraint_error, lambda0, method='Nelder-Mead', tol=1e-5)
            lam = sol.x

            ell = self.ell_mean_std(lam, m)
            self.G_Q_inv = self.get_iso(ell)

            mean, std = self.get_mean_std_stressed()

            if not (np.abs(mean - m) > 1e-4 or np.abs(std - s) > 1e-4):
                search = False

        mean_P, std_P = self.get_mean_std_baseline()

        print("lambda = ", lam)
        print(" WD = ", self.wasserstein_distance(), end="\n")
        print(" Mean, Std = ", mean, std, end="\n")
        print(" Targets = ", m, s, end="\n")
        print(" Base Mean, Std = ", mean_P, std_P, end="\n")
        print("\n")

        fig = self.plot_ell_iso(ell, title)

        return lam, self.wasserstein_distance(), [mean, std], fig

    def optimise_rm_mean_std(self, rm, m, s, title=""):
        if not self.gammas:
            print("Please set gamma functions before calling optimise.")
            return

        def constraint_error(lam):
            # Get the stressed inverse distribution G_Q_inv
            ell = self.ell_rm_mean_std(lam, m)
            G_Q_inv = self.get_iso(ell)

            # Get stressed risk measure RM, mean and standard deviation and calculate the error to minimize
            RM = self.get_risk_measure(G_Q_inv)
            mean, std = self.get_mean_std(G_Q_inv)
            error = np.sqrt(2*(mean - m) ** 2 + 2*(std - s) ** 2 + np.sum((rm - RM) ** 2) / len(self.gammas))

            return error

        search = True
        while search:

            lambda0 = np.random.normal(size=2+len(rm))

            sol = optimize.minimize(constraint_error, lambda0, method='Nelder-Mead', tol=1e-5)
            lam = sol.x

            ell = self.ell_rm_mean_std(lam, m)
            self.G_Q_inv = self.get_iso(ell)

            RM = self.get_risk_measure_stressed()
            mean, std = self.get_mean_std_stressed()

            if not (np.abs(mean - m) > 1e-4 or np.abs(std - s) > 1e-4 or (np.abs(RM - rm) > 1e-4).any()):
                search = False

        RM_P = self.get_risk_measure_baseline()
        mean_P, std_P = self.get_mean_std_baseline()

        print("lambda = ", lam)
        print(" WD = ", self.wasserstein_distance(), end="\n")
        print(" RM = ", RM, end="\n")
        print(" Target Risk Measures = ", rm, end="\n")
        print(" Base Risk Measure = ", RM_P, end="\n")
        print(" Mean, Std = ", mean, std, end="\n")
        print(" Target Mean, Std = ", m, s, end="\n")
        print(" Base Mean, Std = ", mean_P, std_P, end="\n")
        print("\n")

        fig = self.plot_ell_iso(ell, title)

        return lam, self.wasserstein_distance(), RM, [mean, std], fig

    def optimise_HARA(self, a, b, eta, c, rm):

        self.iter = 0

        def constraint_error(lam):

            ell = self.ell_rm(lam[1:])
            iso_g = self.get_iso(ell)

            G_Q_inv = self.UTransform(a, b, eta, self.u, iso_g, np.exp(lam[0]))

            RM = self.get_risk_measure(G_Q_inv)
            Utility = self.get_hara_utility(a, b, eta, self.u, G_Q_inv)

            self.iter += 1
            if np.mod(self.iter, 50) == 0:
                print(lam, RM, Utility)

            error = np.sqrt(np.mean((RM - rm) ** 2) + (Utility - c) ** 2)

            return error

        search = True

        while search:

            lambda0 = np.random.normal(size=len(self.gammas) + 1)

            sol = optimize.minimize(constraint_error, lambda0, method='Nelder-Mead', tol=1e-5)
            lam = sol.x

            ell = self.ell_rm(lam[1:])
            iso_g = self.get_iso(ell)

            self.G_Q_inv = self.UTransform(a, b, eta, self.u, iso_g, np.exp(lam[0]))

            RM = self.get_risk_measure_stressed()
            Utility = self.get_hara_utility(a, b, eta, self.u, self.G_Q_inv)

            if not ((np.abs(RM - rm) > 1e-4).any() or (np.abs(Utility - c) > 1e-4)):
                search = False

        lam_actual = lam.copy()
        lam_actual[0] = np.exp(lam[0])

        print("lambda = ", lam_actual)
        print(" WD = ", self.wasserstein_distance(), end="\n")
        print(" RM, Utility = ", RM, Utility, end="\n")
        print(" Targets = ", rm, c, end="\n")
        print(" Base = ", self.get_risk_measure_baseline(), self.get_hara_utility(a, b, eta, self.u, self.G_P_inv), end="\n")
        print("\n")

        fig = self.plot_ell_iso(ell)
        return lam_actual, self.wasserstein_distance(), [RM, Utility], fig
