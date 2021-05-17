# -*- coding: utf-8 -*-
"""
created on Tue Mar  2 13:51:39 2021

@author: sebja
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from W_Stress_data import W_Stress


# -------------------- Helper Functions -------------------- #
def integrate(f, x):
    return np.sum(0.5 * (f[:-1] + f[1:]) * np.diff(x))


def create_a_b_grid(a, b, N):
    eps = 0.002

    u_eps = 10 ** (np.linspace(-10, np.log(eps) / np.log(10), 10)) - 1e-11
    u_eps_flip = np.flip(copy.deepcopy(u_eps))

    u1 = a + u_eps
    u2 = np.linspace(a + eps, b - eps, N)
    u3 = b - u_eps_flip

    return np.concatenate((u1, u2, u3))


def create_u_grid(pts):
    eps = 1e-5

    knots = np.sort(pts)

    u = create_a_b_grid(eps, knots[0], 100)
    for i in range(1, len(knots)):
        u = np.concatenate((u, create_a_b_grid(knots[i - 1], knots[i], 500)))

    u = np.concatenate((u, create_a_b_grid(knots[-1], 1 - eps, 100)))

    return u


def plot_dist(StressModel, filename, f, F, type="", title="", save=True):
    y_P = np.linspace(0.01, 10, 500)
    x_Q = np.linspace(StressModel.Gs_inv[3], StressModel.Gs_inv[-3], 1000)
    _, gs, Gs = StressModel.distribution(StressModel.u, StressModel.Gs_inv, x_Q)

    fig = plt.figure(figsize=(5, 4))
    plt.plot(x_Q, gs, color='r', label='$g^*_Y$')
    plt.plot(y_P, f(y_P), '--', color='b', label='$f_Y$')

    plt.ylim(0, 0.6)
    plt.xlim(0, 10)
    plt.title(title)
    plt.legend(fontsize=14)

    plt.show()

    if save:
        fig.savefig(filename + '_density.pdf', format='pdf')

    fig = plt.figure(figsize=(5, 4))
    plt.plot(x_Q, Gs, color='r', label='$G^*_Y$')
    plt.plot(y_P, F(y_P), '--', color='b', label='$F_Y$')
    plt.xlim(0, 10)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=14)
    plt.title(title)
    plt.show()

    if save:
        fig.savefig(filename + '_CDF.pdf', format='pdf')

    mask = (np.diff(StressModel.Gs_inv) < 1e-10)
    if np.sum(mask) > 0:
        idx = np.where(mask)[0][0]
    else:
        idx = []

    fig = plt.figure(figsize=(4, 4))

    dQ_dP = gs / f(x_Q)
    if type == "ES":
        dQ_dP[:idx] = 1
    plt.plot(x_Q, dQ_dP)

    plt.ylim(0, 2)
    plt.xlabel(r"$y$", fontsize=18)
    plt.ylabel(r"$\frac{g^{*}_Y}{f_Y}$", fontsize=22)
    plt.title(title)
    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig(filename + '_RN.pdf', format='pdf')


if __name__ == "__main__":
    # -------------------- Generate data -------------------- #
    np.random.seed(0)
    Nsims = 100
    p_mix = 0.2
    H = (np.random.uniform(size=Nsims) < p_mix)
    # data = {"y": np.random.normal(loc=7 * H + (1 - H) * 5, scale=(1 + H) / 5)}
    data = {"y": np.random.lognormal(mean=1-0.5*0.5**2, sigma=0.5, size=Nsims)}

    h_y = 1.06 * np.std(data["y"]) * (len(data["y"])) ** (-1 / 5) / 2
    f = lambda y: np.sum(norm.pdf((y.reshape(-1, 1) - data["y"].reshape(1, -1)) / h_y) / h_y / len(data["y"]),
                         axis=1).reshape(-1)

    F = lambda y: np.sum(norm.cdf((y.reshape(-1, 1) - data["y"].reshape(1, -1)) / h_y) / len(data["y"]), axis=1).reshape(-1)


    # -------------------- ES_0 and ES_0.95 -------------------- #
    alpha = [0, 0.95]

    y = np.linspace(1e-20, 30, 1000)
    u = create_u_grid([0.95])

    # -------------------- Generate the model -------------------- #
    StressModel = W_Stress(data, u)

    # -------------------- Optimize ES risk measure -------------------- #
    gammas = [lambda u: (u > alpha[0]) / (1 - alpha[0]), lambda u: (u > alpha[1]) / (1 - alpha[1])]

    # for gamma in gammas:
    #     plt.plot(u, gamma(u))
    # plt.ylabel(r'$\gamma(u)$')
    # plt.xlabel('u')
    # plt.show()

    # Set gammas
    StressModel.set_gamma(gammas)

    RM_P = StressModel.get_risk_measure_baseline()
    print(RM_P)

    es_stresses = [[5, 5],
                   [-5, 5]]
    for es_stress in es_stresses:

        lam, WD, RM_Q, fig = StressModel.optimise_rm(RM_P * np.array([1+es_stress[0]/100,
                                                                      1+es_stress[1]/100]))
        filename = f'Plots/1D/ES/data_ES0_{es_stress[0]}_ES95_{es_stress[1]}'

        fig.savefig(filename + '_inv.pdf', format='pdf')
        plot_dist(StressModel, filename, f, F, "ES", save=True)

    # -------------------- Test Mean and Variance Optimisation -------------------- #
    mean_P, std_P = StressModel.get_mean_std(StressModel.F_inv)
    lam, WD, mv_Q, fig = StressModel.optimise_mean_std(mean_P, 1.2 * std_P)

    filename = 'Plots/1D/mean-std/data_M_S_20'
    fig.savefig(filename + '_inv.pdf',format='pdf')

    plot_dist(StressModel, filename, f, F, "meand-std")

    # -------------------- Test Utility and risk measure -------------------- #
    # ******** NOT Converging ********

    hara = lambda a, b, eta, x: (1 - eta) / eta * (a * x / (1 - eta) + b) ** eta

    b = lambda eta: 5 * (eta / (1 - eta)) ** (1 / eta)
    plt.plot(y, hara(1, b(0.2), 0.2, y))

    RM_P = StressModel.get_risk_measure_baseline()
    Utility_P = StressModel.get_hara_utility(1, b(0.2), 0.2, StressModel.u, StressModel.F_inv)

    _, _, _, fig = StressModel.optimise_HARA(1, b(0.2), 0.2, Utility_P * 1, RM_P * np.array([1.0, 1.0]))

    filename = 'Plots/1D/HARA-ES/data_utility_rm_1_0_95_s5_downup'
    fig.savefig(filename + '_inv.pdf', format='pdf')

    plot_dist(StressModel, filename, f, F)
