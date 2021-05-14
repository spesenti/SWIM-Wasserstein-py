import copy
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.stats import norm

from W_Stress import W_Stress


# -------------------- Helper Functions -------------------- #
def integrate(f, x):
    # integral of f over space x using Reimann sums
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


def plot_dist(StressModel, filename, f, F, type="", title="", save=True, rm=None):
    y_P = np.linspace(0.01, 10, 500)
    x_Q = np.linspace(StressModel.Gs_inv[2], 10, 1000)
    _, gs, Gs = StressModel.distribution(StressModel.u, StressModel.Gs_inv, x_Q)

    fig = plt.figure(figsize=(5, 4))
    plt.plot(x_Q, gs, color='r', label='$g^*_Y$')
    plt.plot(y_P, f(y_P), '--', color='b', label='$f_Y$')

    if type == "rm-mean-std" and rm is not None:
        colors = ["r", "b"]
        labels = ["$ES_{0.95}(G^*_Y)$", "$ES_{0.95}(F_Y)$"]
        for i in range(len(rm)):
            plt.axvline(x=rm[i], color=colors[i], linestyle='dotted', label=labels[i])

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
    # -------------------- ES_0.8 and ES_0.95 with log-normal(mu, sigma) -------------------- #
    # log-normal params
    mu = 1
    sigma = 0.5

    # collection of ES_alpha levels
    alpha = [0.8, 0.95]

    # grids in random variable space (0, inf), and in quantile space (0, 1)
    y = np.linspace(1e-20, 30, 1000)
    u = create_u_grid(alpha)

    # log-normal pdf and cdf
    f = lambda y: norm.pdf((np.log(y) - (mu - 0.5 * sigma ** 2)) / sigma) / (y * sigma)
    F = lambda y: norm.cdf((np.log(y) - (mu - 0.5 * sigma ** 2)) / sigma)

    # -------------------- Generate the model -------------------- #
    StressModel = W_Stress(y, F, u)

    plt.plot(u, StressModel.F_inv)
    plt.plot(F(y), y)
    plt.yscale('log')
    plt.ylim(1e-3, 1e2)
    plt.show()

    # -------------------- Optimize ES risk measure -------------------- #
    # this is a collection of lambda's that encode the constraints
    gammas = [lambda u: (u > alpha[0]) / (1 - alpha[0]), lambda u: (u > alpha[1]) / (1 - alpha[1])]

    for gamma in gammas:
        plt.plot(u, gamma(u))
    plt.ylabel(r'$\gamma(u)$')
    plt.xlabel('u')
    plt.show()

    # Set gammas
    StressModel.set_gamma(gammas)

    # compute the risk-measure of the base model
    RM_P = StressModel.get_risk_measure_baseline()

    lam, WD, RM_Q, fig = StressModel.optimise_rm(RM_P * np.array([1.1, 1.1]))

    filename = 'Plots/lognormal_ES_80_95_s10'

    fig.savefig(filename + '_inv.pdf', format='pdf')

    plot_dist(StressModel, filename, f, F, "ES")

    # -------------------- Optimize alpha-beta risk measure -------------------- #
    p_list = [0.25, 0.5, 0.75]
    alpha = 0.9
    beta = 0.1

    for p in p_list:
        alpha_beta_gamma = [lambda u: ((u < beta) * p + (u >= alpha) * (1 - p)) / (p * beta + (1 - p) * (1 - alpha))]

        plt.plot(u, alpha_beta_gamma[0](u))
        plt.ylabel(r'$\alpha_{0.9}-\beta_{0.1} \gamma(u)$')
        plt.xlabel('u')
        plt.show()

        StressModel.set_gamma(alpha_beta_gamma)

        # compute the baseline risk-measure of the base model
        RM_P = StressModel.get_risk_measure_baseline()
        lam, WD, RM_Q, fig = StressModel.optimise_rm(RM_P * np.array([1.1]), title=f"p={p}")

        filename = f'Plots/alpha-beta/lognormal_alpha_{alpha}_beta_{beta}_p_{p}_s10'

        fig.savefig(filename + '_inv.pdf', format='pdf')

        plot_dist(StressModel, filename, f, F, "ES", title=f"p={p}")

    # -------------------- Test Mean and Variance Optimisation -------------------- #
    mean_P, std_P = StressModel.get_mean_std_baseline()

    lam, WD, mv_Q, fig = StressModel.optimise_mean_std(mean_P, 1.2 * std_P)

    filename = 'Plots/lognormal_M_S_20'
    fig.savefig(filename + '_inv.pdf', format='pdf')

    plot_dist(StressModel, filename, f, F, "mean-std")

    # -------------------- Test mean-variance + ES measure -------------------- #

    alpha = 0.95
    gamma_ES = [lambda u: (u >= alpha) / (1 - alpha)]

    StressModel.set_gamma(gamma_ES)

    RM_P = StressModel.get_risk_measure_baseline()
    mean_P, std_P = StressModel.get_mean_std_baseline()

    rm_stresses = [10, 10, 0]
    mean_stresses = [10, -10, 0]
    std_stresses = [-10, 0, 20]

    # stresses = [-10, 0, 10]
    # for stress in itertools.product(stresses, repeat=3):

    for i in range(len(rm_stresses)):
        rm_stress = rm_stresses[i]
        mean_stress = mean_stresses[i]
        std_stress = std_stresses[i]
        _, _, RM_Q, _, fig = StressModel.optimise_rm_mean_std(np.array([1 + rm_stress / 100]) * RM_P,
                                                              (1 + mean_stress / 100) * mean_P,
                                                              (1 + std_stress / 100) * std_P)

        filename = f'Plots/ES-mean-std/lognormal_alpha_{alpha}_ES_{rm_stress}_M_{mean_stress}_S_{std_stress}'
        fig.savefig(filename + '_inv.pdf', format='pdf')

        plot_dist(StressModel, filename, f, F, "rm-mean-std", rm=[RM_Q[0], RM_P[0]])

    # -------------------- Test Utility and risk measure -------------------- #
    hara = lambda a, b, eta, x: (1 - eta) / eta * (a * x / (1 - eta) + b) ** eta

    b = lambda eta: 5 * (eta / (1 - eta)) ** (1 / eta)
    x = np.linspace(0.001, 2, 100)
    plt.plot(x, hara(1, b(0.2), 0.2, x))

    # Set gammas
    alpha = [0.8, 0.95]
    gammas = [lambda u: (u > alpha[0]) / (1 - alpha[0]), lambda u: (u > alpha[1]) / (1 - alpha[1])]
    StressModel.set_gamma(gammas)

    RM_P = StressModel.get_risk_measure_baseline()
    Utility_P = StressModel.get_hara_utility(1, b(0.2), 0.2, StressModel.u, StressModel.F_inv)

    utility_stresses = [0, 1, 3]
    rm_stresses = [-10, 10]

    for utility_stress in utility_stresses:
        _, _, _, fig = StressModel.optimise_HARA(1, b(0.2), 0.2, Utility_P * (1 + utility_stress / 100),
                                                 RM_P * np.array([1 + rm_stresses[0] / 100, 1 + rm_stresses[1] / 100]))

        filename = f'Plots/HARA-ES/lognormal_utility_{utility_stress}_ES_{rm_stresses[0]}_{rm_stresses[1]}'
        fig.savefig(filename + '_inv.pdf', format='pdf')

        plot_dist(StressModel, filename, f, F, "Utility")
