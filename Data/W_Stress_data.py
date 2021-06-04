# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:37:12 2021

@author: sebja
"""

import numpy as np

from sklearn.isotonic import IsotonicRegression

from scipy import interpolate
from scipy import optimize

from scipy.stats import norm

import matplotlib.pyplot as plt


class W_Stress:

    def __init__(self, data, u, bracket=None):
        self.ir = IsotonicRegression()
        self.data = data
        self.u = u

        nY = 500
        y = np.linspace(0.9 * np.quantile(data["y"], 0.005), np.quantile(data["y"], 0.995) * 1.1, nY)
        self.y = y

        self.h_y = 1.06 * np.std(data["y"]) * (len(data["y"])) ** (-1 / 5)
        self.h_x = 1.06 * np.std(data["x"], axis=0) * (len(data["y"])) ** (-1 / 5)
        self.f = lambda y: np.sum(norm.pdf((y.reshape(-1, 1) - data["y"].reshape(1, -1)) / self.h_y) /
                                  self.h_y / len(data["y"]), axis=1).reshape(-1)

        self.F = lambda y: np.sum(norm.cdf((y.reshape(-1, 1) - data["y"].reshape(1, -1)) / self.h_y) / len(data["y"]),
                                  axis=1).reshape(-1)

        if not bracket:
            bracket = [-10, 20]
        F_inv = lambda u: optimize.root_scalar(lambda y: (self.F(np.array([y])) - u), method='Brentq',
                                               bracket=bracket)

        self.F_inv = np.zeros(len(u))
        for i in range(len(u)):
            self.F_inv[i] = F_inv(u[i]).root

        # Initialize
        self.Gs_inv = None
        self.gammas = []
        self.gs = None
        self.Gs = None

    # Static functions
    @staticmethod
    def integrate(f, u):
        # integral of f over space x using Reimann sums
        return np.sum(0.5 * (f[:-1] + f[1:]) * np.diff(u))

    # Plot functions
    def plot_G_inv(self, G_inv, title=""):

        fig = plt.figure(figsize=(4, 4))
        plt.plot(self.u, G_inv, label=r"$\breve{G}^*_Y$", color='r')
        plt.plot(self.u, self.F_inv, linestyle='--', color='b', label=r"$\breve{F}_Y$")
        plt.title(title)
        plt.legend(fontsize=14)

        plt.yscale('log')

        plt.show()

        return

    def plot_ell_iso(self, ell, title=""):

        fig = plt.figure(figsize=(4, 4))
        plt.plot(self.u, ell, linewidth=0.5, color='g', label=r"$\ell$")
        plt.plot(self.u, self.Gs_inv, label=r"$\breve{G}^*_Y$", color='r')
        plt.plot(self.u, self.F_inv, linestyle='--', color='b', label=r"$\breve{F}_Y$")
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
        plt.plot(self.u, self.Gs_inv, linestyle='--', label=r"$\breve{G}^*_Y$", color='r')
        plt.legend(fontsize=14)
        plt.yscale('log')

        plt.show()

        return fig

    def plot_f_F(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        ax2 = ax[0].twinx()

        ax[0].plot(self.y, self.f(self.y), color='k')
        ax2.plot(self.y, self.F(self.y), color='r')

        ax[0].set_xlabel(r"$y$", fontsize=16)
        ax[0].set_ylabel(r"$f(y)$", fontsize=16)
        ax2.set_ylabel(r"$F(y)$", fontsize=16, color='r')
        ax2.set_ylim(0, 1.02)

        ax[1].plot(self.u, self.F_inv)
        ax[1].set_ylabel(r"$F^{-1}(y)$", fontsize=16)
        ax[1].set_xlabel(r"$y$", fontsize=16)

        plt.tight_layout(pad=1)

        plt.show()
        return

    def plot_sensitivities(self, sensitivity_measures, filename, colors, labels, title=None, save=True):
        n = len(sensitivity_measures)
        w = round(1/n, 1) - 0.1

        scale = - np.floor(n/2)
        for i in range(n):
            if n%2 == 0:
                plt.bar(np.arange(10) + scale * w, sensitivity_measures[i], color=colors[i],
                    label=labels[i], width=w)
                scale += 1

            else:
                plt.bar(range(10) + scale * w, sensitivity_measures[i], color=colors[i],
                        label=labels[i], width=w)
                scale += 1

        plt.title(title)
        plt.xlabel('X')
        plt.ylim(-1,1)
        plt.legend()

        if save:
            plt.savefig(filename, type='pdf')

        plt.show()

    def plot_dist(self, filename, type="", title="", save=True):
        y_P = np.linspace(self.F_inv[5], self.F_inv[-5], 1000)
        y_Q = np.linspace(self.Gs_inv[3], self.Gs_inv[-3], 1000)

        self.set_distribution(self.u, self.Gs_inv, y_Q)

        fig = plt.figure(figsize=(5, 4))
        plt.plot(y_Q, self.gs(y_Q), color='r', label='$g^*_Y$')
        plt.plot(y_P, self.f(y_P), '--', color='b', label='$f_Y$')
        plt.ylim(bottom=0)
        plt.title(title)
        plt.show()

        if save:
            fig.savefig(filename + '_density.pdf', format='pdf')

        fig = plt.figure(figsize=(5, 4))
        plt.plot(y_Q, self.Gs(y_Q), color='r', label='$G^*_Y$')
        plt.plot(y_P, self.F(y_P), '--', color='b', label='$F_Y$')
        plt.legend(fontsize=14)
        plt.title(title)
        plt.show()

        if save:
            fig.savefig(filename + '_CDF.pdf', format='pdf')

        idx = np.where(np.diff(self.Gs_inv) < 1e-8)[0][0]

        fig = plt.figure(figsize=(4, 4))

        dQ_dP = self.gs(y_Q) / self.f(y_Q)
        if type == "ES":
            dQ_dP[:idx] = 1
        plt.plot(y_Q, dQ_dP)

        plt.ylim(0, 15)
        plt.xlabel(r"$y$", fontsize=18)
        plt.ylabel(r"$\frac{g^{*}_Y}{f_Y}$", fontsize=22)
        plt.tight_layout()
        plt.show()

        if save:
            fig.savefig(filename + '_RN.pdf', format='pdf')

        # Get index for x- and y-values
        idx = (self.data["y"] >= y_Q[0]) & (self.data["y"] <= y_Q[-1])

        # Get adjusted g*/f
        w = self.get_weights()

        data_y_idx = self.data["y"][idx]
        data_x_idx = self.data["x"][idx]
        w_idx = w[idx]

        num_inputs = self.data['x'].shape[1]
        nrows = int(np.ceil(num_inputs/2))
        fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(15,10))

        ind = 0
        for i in range(nrows):
            for j in range(2):
                x_axis = np.linspace(0.9 * np.quantile(self.data["x"][:, ind], 0.005),
                                     np.quantile(self.data["x"][:, ind], 0.995) * 1.1,
                                     len(w))

                f_x = self.f_x(self.data["x"][:, ind], self.h_x[ind])  # under P: KDE of marginal distribution of xi
                gs_x = self.gs_x(self.data["x"][:, ind], self.h_x[ind], w)  # under Q: KDE of marginal distribution of xi

                mean_f = np.mean(self.data["x"][:, ind])
                mean_gs = np.sum(self.data["x"][:, ind] * w) / np.sum(w)

                ax[i, j].plot(x_axis, f_x(x_axis), color='blue', label='$f_X$')
                ax[i, j].plot(x_axis, gs_x(x_axis), color='red', label='$g^*_X$')
                ax[i, j].axvline(x=mean_f, color='blue', linestyle='dashed', label='E[$f_X$]')
                ax[i, j].axvline(x=mean_gs, color='red', linestyle='dashed', label='E[$g^*_X$]')
                ax[i, j].set_title('$X_{%d}$'%(ind+1))

                ind += 1

        plt.legend()
        fig.tight_layout()
        plt.show()

        return

    # Get statistics (Wasserstein distance, risk measure, mean, standard deviation)
    def wasserstein_distance(self):
        # Calculate the Wasserstein distance W2(G, F)
        return np.sqrt(self.integrate((self.Gs_inv - self.F_inv) ** 2, self.u))

    def reverse_sensitivity_measure(self, s, x):
        """
        :param s: lambda function s:R -> R
        :return: array of reverse sensitivity measures
        """

        if self.Gs_inv is None:
            print("No stressed model defined. Please run an optimization before proceeding.")
            return None

        w = self.get_weights()
        N = x.shape[0]

        if x.ndim == 1:
            EQ_sX = np.sum(np.multiply(s(x), w)) / N
            EP_sX = np.mean(s(x))

            x_inc = np.sort(x)
            if EQ_sX >= EP_sX:
                w_inc = np.sort(w)
                max_EQ = np.mean(np.multiply(s(x_inc), w_inc))
                S = (EQ_sX - EP_sX) / (max_EQ - EP_sX)

            else:
                w_dec = np.sort(w)[::-1]
                min_EQ = np.mean(np.multiply(s(x_inc), w_dec))
                S = -(EQ_sX - EP_sX) / (min_EQ - EP_sX)

        else:
            num_inputs = x.shape[1]
            S = np.zeros(num_inputs)

            for i in range(num_inputs):
                EQ_sX = np.sum(np.multiply(s(x[:,i]) , w)) / N
                EP_sX = np.mean(s(x[:,i]))

                xi_inc = np.sort(x[:, i])
                if EQ_sX >= EP_sX:
                    w_inc = np.sort(w)
                    max_EQ = np.mean(np.multiply(s(xi_inc), w_inc))
                    S[i] = (EQ_sX - EP_sX)/(max_EQ - EP_sX)

                else:
                    w_dec = np.sort(w)[::-1]
                    min_EQ = np.mean(np.multiply(s(xi_inc), w_dec))
                    S[i] = -(EQ_sX - EP_sX) / (min_EQ - EP_sX)

        return S

    def get_weights(self):
        # Get smoother results, dependent on KDE approximation
        y_gd = np.linspace(self.Gs_inv[3], self.Gs_inv[-3], 500)

        # Get gs, Gs
        self.set_distribution(self.u, self.Gs_inv, y_gd)

        # print("computing gs at grid points...")
        gs = self.gs(y_gd)
        gs /= self.integrate(gs, y_gd)

        # print("computing f at grid points...")
        f = self.f(y_gd)
        f /= self.integrate(f, y_gd)

        dQ_dP = gs / f

        # print("E[dQ/dP]", self.integrate(dQ_dP * f, y_gd))

        # print("computing weights...")
        w = np.zeros(len(self.data["y"]))
        for i in range(len(self.data["y"])):
            w[i] = self.integrate(norm.pdf((y_gd - self.data["y"][i]) / self.h_y) / self.h_y * dQ_dP, y_gd)

        # Normalize w to sum to N = len(self.data["y"])
        w = w/np.sum(w) * len(self.data["y"])

        return w

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
        return self.get_risk_measure(self.F_inv)

    def get_risk_measure_stressed(self):
        if self.Gs_inv is None:
            print("Stressed distribution does not exist.")
            return None
        else:
            return self.get_risk_measure(self.Gs_inv)

    def get_mean_std(self, G_inv):

        # Calculate mean of distribution E[X] = int{1 - G(x) dx} = int{G_inv(u) du}
        mean = self.integrate(G_inv, self.u)

        # Calculate standard deviation of distribution
        std = np.sqrt(self.integrate((G_inv - mean) ** 2, self.u))

        return mean, std

    def get_mean_std_baseline(self):
        return self.get_mean_std(self.F_inv)

    def get_mean_std_stressed(self):
        if self.Gs_inv is None:
            print("Stressed distribution does not exist.")
            return None
        else:
            return self.get_mean_std(self.Gs_inv)

    def get_hara_utility(self, a, b, eta, u, G_inv):
        # f = (1 - eta) / eta * (a * G_inv / (1 - eta) + b) ** eta
        dummy = a * G_inv / (1 - eta) + b
        f = (1 - eta) / eta * np.sign(dummy) * (np.abs(dummy)) ** eta
        return self.integrate(f, u)

    def UTransform(self, a, b, eta, u, G_inv, lam):

        g = np.zeros(len(u))

        # nu(x) = x - lam * u'(x)
        # u(x) = (1 - eta)/eta * (a * x / (1-eta) + b)^eta
        # u'(x) = a * (a * x / (1-eta) + b)^(eta - 1)
        nu = lambda x: x - lam * a * (a / (1 - eta) * x + b) ** (eta - 1)

        # Get g = nu_inv(.)
        plot = False
        last_g = 0
        exception = False
        for i in range(len(u)):
            try:
                g[i] = optimize.root_scalar(lambda x: nu(x) - G_inv[i], method='bisect',
                                            bracket=[-b * (1 - eta) / a + 1e-10, 500]).root
                last_g = g[i]
            except Exception as e:
                g[i] = last_g
                exception = True

        if exception:
            print("Exception found at", last_g)
            # self.plot_G_inv(G_inv)
            # x = np.linspace(-b * (1 - eta) / a + 1e-10, 100, 1000)
            # plt.plot(x, nu(x))
            # plt.yscale('log')
            # plt.show()

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
        ell = self.F_inv.copy()

        for i in range(len(self.gammas)):
            ell += lam[i] * self.gammas[i]

        return ell

    def ell_mean_std(self, lam, m):
        # ell = 1/(1 + lambda2) * (F_inv(u) + lambda_1 + lambda_2 * m)
        ell = (self.F_inv + lam[0] + lam[1] * m) / (1 + lam[1])

        return ell

    def ell_rm_mean_std(self, lam, m):
        # ell = 1/(1 + lambda2) * (F_inv(u) + lambda_1 + lambda_2 * m + sum{lambda_{k+2} + gamma_k(u)})

        ell_partial = self.ell_rm(lam[2:])  # ell_partial = F_inv(u) + sum{lambda_{k+2} * gamma_k(u)}
        ell = (lam[0] + lam[1] * m + ell_partial) / (1 + lam[1])

        return ell

    def get_iso(self, ell):
        return self.ir.fit_transform(self.u, ell)

    # distribution
    def set_distribution(self, u, G_inv, x):

        # print("using qtl derivative")

        eps = np.cumsum(1e-10 * np.ones(len(G_inv)))
        x_coarse = eps + G_inv

        # print("G coarse | ", np.min(x_coarse), np.max(x_coarse))
        # print("G fine | ", np.min(x), np.max(x))

        if np.min(x_coarse) > np.min(x):
            # Manually fix interpolation range
            x_coarse[0] = np.min(x)

        G_interp = interpolate.interp1d(x_coarse, u, kind='linear', fill_value='extrapolate')

        # print("G_inv coarse | ", np.min(0.5 * (u[2:] + u[:-2])), np.max(0.5 * (u[2:] + u[:-2])))
        # print("G_inv fine | ", np.min(G_interp(x)), np.max(G_interp(x)))

        x_inv_coarse = 0.5 * (u[2:] + u[:-2])
        if np.min(np.min(x_inv_coarse) > np.min(G_interp(x))):
            # Manually fix interpolation range
            x_inv_coarse[0] = np.min(G_interp(x))


        dG_inv = (G_inv[2:] - G_inv[:-2]) / (u[2:] - u[:-2])
        dG_inv_interp = interpolate.interp1d(x_inv_coarse, dG_inv, kind='linear', fill_value='extrapolate')

        G = G_interp(x)
        g = 1 / dG_inv_interp(G_interp(x))

        self.Gs = G_interp
        self.gs = lambda x: 1 / dG_inv_interp(G_interp(x))

        return

    def f_x(self, xi, h_xi):
        """
        :param x_grid: 1D array representing points for evaluation
        :param xi: 1D array representing xi
        :return f: function representing KDE density of xi_data under P evaluated at x_grid
        """

        f = lambda x: np.sum(norm.pdf((x.reshape(-1, 1) - xi.reshape(1, -1)) / h_xi) /
                             h_xi / len(xi), axis=1).reshape(-1)

        return f

    def gs_x(self, xi, h_xi, w):
        """
        :param f_x: 1D array representing density under P evaluated at x_grid
        :param w: 1D array representing the weights
        :return gs: 1D representing KDE density of xi_data under Q evaluted at x_grid
        """

        gs = lambda x: np.sum(w * norm.pdf((x.reshape(-1, 1) - xi.reshape(1, -1)) / h_xi) /
                             h_xi / len(xi), axis=1).reshape(-1)

        return gs

    # optimise functions
    def optimise_rm(self, rm, title=""):
        # Solve optimization problem with risk measure constraints gammas
        # min. W2(G,F) s.t. rho_gamma(G) = r for each risk measure

        if not self.gammas:
            print("Please set gamma functions before calling optimise.")
            return

        self.iter = 0
        # Calculate the error for set of Lagrange multipliers lambda
        def constraint_error(lam):
            # Get the stressed inverse distribution Gs_inv
            ell = self.ell_rm(lam)
            Gs_inv = self.get_iso(ell)

            # Get stressed risk measure RM and calculate the error to minimize
            RM = self.get_risk_measure(Gs_inv)
            RM_Error = np.sqrt(np.sum((rm - RM) ** 2) / len(self.gammas))

            return RM_Error

        search = True

        while search:

            lambda0 = np.random.normal(size=(len(rm)))

            sol = optimize.minimize(constraint_error, lambda0, method='Nelder-Mead', tol=1e-5)
            lam = sol.x

            ell = self.ell_rm(lam)
            self.Gs_inv = self.get_iso(ell)

            if self.iter > 1000 * 50:
                # manually terminate search
                search = False
                print("WARNING: Search incomplete. Terminating ... ")

            if not (np.abs(rm - self.get_risk_measure_stressed()) > 1e-4).any():
                search = False

        print("lambda = ", lam)
        print(" WD = ", self.wasserstein_distance(), end="\n")
        print(" Risk Measure = ", self.get_risk_measure_stressed(), end="\n")
        print(" Target Risk Measure = ", rm, end="\n")
        print(" Base Risk Measure = ", self.get_risk_measure_baseline(), end="\n")
        print("\n")

        fig = self.plot_ell_iso(ell, title)
        fig = self.plot_ell(ell)
        return lam, self.wasserstein_distance(), self.get_risk_measure_stressed(), fig

    def optimise_mean_std(self, m, s, title=""):

        self.iter = 0
        def constraint_error(lam):

            ell = self.ell_mean_std(lam, m)
            Gs_inv = self.get_iso(ell)
            mean, std = self.get_mean_std(Gs_inv)
            error = np.sqrt(2) * np.sqrt((mean - m) ** 2 + (std - s) ** 2)  # sqrt(2) normalization constant

            return error

        search = True

        while search:

            lambda0 = np.random.normal(size=2)

            sol = optimize.minimize(constraint_error, lambda0, method='Nelder-Mead', tol=1e-5)
            lam = sol.x

            ell = self.ell_mean_std(lam, m)
            self.Gs_inv = self.get_iso(ell)

            mean, std = self.get_mean_std_stressed()

            if self.iter > 1000 * 50:
                # manually terminate search
                search = False
                print("WARNING: Search incomplete. Terminating ... ")

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

        self.iter = 0
        def constraint_error(lam):
            # Get the stressed inverse distribution Gs_inv
            ell = self.ell_rm_mean_std(lam, m)
            Gs_inv = self.get_iso(ell)

            # Get stressed risk measure RM, mean and standard deviation and calculate the error to minimize
            RM = self.get_risk_measure(Gs_inv)
            mean, std = self.get_mean_std(Gs_inv)

            self.iter += 1

            error = np.sqrt(2*(mean - m) ** 2 + 2*(std - s) ** 2 + np.sum((rm - RM) ** 2) / len(self.gammas))

            return error

        search = True
        while search:

            lambda0 = np.random.normal(size=2+len(rm))

            sol = optimize.minimize(constraint_error, lambda0, method='Nelder-Mead', tol=1e-5)
            lam = sol.x

            ell = self.ell_rm_mean_std(lam, m)
            self.Gs_inv = self.get_iso(ell)

            RM = self.get_risk_measure_stressed()
            mean, std = self.get_mean_std_stressed()

            if self.iter > 1000 * 50:
                # manually terminate search
                search = False
                print("WARNING: Search incomplete. Terminating ... ")

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

            Gs_inv = self.UTransform(a, b, eta, self.u, iso_g, np.exp(lam[0]))

            RM = self.get_risk_measure(Gs_inv)
            Utility = self.get_hara_utility(a, b, eta, self.u, Gs_inv)

            error = np.sqrt(np.mean((RM - rm) ** 2) + (Utility - c) ** 2)

            self.iter += 1
            if np.mod(self.iter, 1000) == 0:
                print(lam, RM, Utility)

            return error

        search = True

        while search:

            lambda0 = np.random.normal(size=len(self.gammas) + 1)

            sol = optimize.minimize(constraint_error, lambda0, method='Nelder-Mead', tol=1e-5)
            lam = sol.x

            ell = self.ell_rm(lam[1:])
            iso_g = self.get_iso(ell)

            self.Gs_inv = self.UTransform(a, b, eta, self.u, iso_g, np.exp(lam[0]))

            RM = self.get_risk_measure_stressed()
            Utility = self.get_hara_utility(a, b, eta, self.u, self.Gs_inv)

            if self.iter > 1000 * 20:
                # manually terminate search
                search = False
                print("WARNING: Search incomplete. Terminating ... ")

            if not ((np.abs(RM - rm) > 1e-4).any() or (np.abs(Utility - c) > 1e-4)):
                search = False

        lam_actual = lam.copy()
        lam_actual[0] = np.exp(lam[0])

        print("lambda = ", lam_actual)
        print(" WD = ", self.wasserstein_distance(), end="\n")
        print(" RM, Utility = ", RM, Utility, end="\n")
        print(" Targets = ", rm, c, end="\n")
        print(" Base = ", self.get_risk_measure_baseline(), self.get_hara_utility(a, b, eta, self.u, self.F_inv), end="\n")
        print("\n")

        fig = self.plot_ell_iso(ell)

        return lam_actual, self.wasserstein_distance(), [RM, Utility], fig
