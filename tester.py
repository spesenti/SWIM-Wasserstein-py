import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from Data.W_Stress_data import W_Stress as W_Stress_data
from Lognormal.W_Stress import W_Stress as W_Stress_lognormal

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


def plot_density(StressModels, filename, fs, title="", save=True):
    fig = plt.figure(figsize=(5, 4))
    colors = [['r', 'black'], ['b', 'g']]
    labels = ['data', 'lognormal']

    for i in range(len(StressModels)):
        y_P = np.linspace(0.01, 10, 500)
        x_Q = np.linspace(StressModels[i].Gs_inv[3], StressModels[i].Gs_inv[-3], 1000)
        _, gs, Gs = StressModels[i].distribution(StressModels[i].u, StressModels[i].Gs_inv, x_Q)

        plt.plot(x_Q, gs, color=colors[0][i], label=f'$g^*_Y$_{labels[i]}')
        plt.plot(y_P, fs[i](y_P), '--', color=colors[1][i], label=f'$f_Y$_{labels[i]}')

    plt.ylim(0, 0.6)
    plt.xlim(0, 10)
    plt.title(title)
    plt.legend(fontsize=14)

    plt.show()

    if save:
        fig.savefig(filename + '_density.pdf', format='pdf')
    return

def plot_inv(filename, dataModel, lognModel, title="", save=True):
    fig = plt.figure(figsize=(4, 4))
    plt.plot(u, dataModel.Gs_inv, label=r"$\breve{G}^*_Y$ data", color='r')
    plt.plot(u, dataModel.F_inv, linestyle='--', color='b', label=r"$\breve{F}_Y$ data")
    plt.plot(u, lognModel.Gs_inv, label=r"$\breve{G}^*_Y$ lognormal", color='black')
    plt.plot(u, lognModel.F_inv, linestyle='--', color='g', label=r"$\breve{F}_Y$ lognormal")
    plt.title(title)
    plt.legend(fontsize=14)

    plt.yscale('log')

    plt.show()
    if save:
        fig.savefig(filename + '_quantiles.pdf', format='pdf')

    return


if __name__ == "__main__":
    # -------------------- Generate data -------------------- #
    # log-normal params
    mu = 1
    sigma = 0.5

    # data params
    np.random.seed(0)
    Nsims = 100
    p_mix = 0.2
    H = (np.random.uniform(size=Nsims) < p_mix)
    data = {"y": np.random.lognormal(mean=mu-0.5*sigma**2, sigma=sigma, size=Nsims)}

    h_y = 1.06 * np.std(data["y"]) * (len(data["y"])) ** (-1 / 5) / 2
    f_data = lambda y: np.sum(norm.pdf((y.reshape(-1, 1) - data["y"].reshape(1, -1)) / h_y) / h_y / len(data["y"]),
                         axis=1).reshape(-1)

    F_data = lambda y: np.sum(norm.cdf((y.reshape(-1, 1) - data["y"].reshape(1, -1)) / h_y)
                         / len(data["y"]), axis=1).reshape(-1)

    # log-normal pdf and cdf
    f_lognormal = lambda y: norm.pdf((np.log(y) - (mu - 0.5 * sigma ** 2)) / sigma) / (y * sigma)
    F_lognormal = lambda y: norm.cdf((np.log(y) - (mu - 0.5 * sigma ** 2)) / sigma)

    # collection of ES_alpha levels
    alphas = [0.8, 0.95]

    # grids in random variable space (0, inf), and in quantile space (0, 1)
    y = np.linspace(1e-20, 30, 1000)
    u = create_u_grid(alphas)

    # -------------------- Generate the model -------------------- #
    StressModelData = W_Stress_data(data, u)
    StressModelLogN = W_Stress_lognormal(y, F_lognormal, u)

    # -------------------- Optimize alpha-beta risk measure -------------------- #
    p_list = [0.25, 0.5, 0.75]
    alpha = 0.9
    beta = 0.1

    for p in p_list:
        alpha_beta_gamma = [lambda u: ((u < beta) * p + (u >= alpha) * (1 - p)) / (p * beta + (1 - p) * (1 - alpha))]

        StressModelData.set_gamma(alpha_beta_gamma)
        StressModelLogN.set_gamma(alpha_beta_gamma)

        # compute the baseline risk-measure of the base model
        RM_P_data = StressModelData.get_risk_measure_baseline()
        _, WD_data, _, fig_data = StressModelData.optimise_rm(RM_P_data * np.array([1.1]), title=f"p={p}")

        # compute the baseline risk-measure of the base model
        RM_P_logn = StressModelData.get_risk_measure_baseline()
        _, WD_logn, _, fig_logn = StressModelLogN.optimise_rm(RM_P_logn * np.array([1.1]), title=f"p={p}")

        filename = f'Plots/data_lognormal_alpha_{alpha}_beta_{beta}_p_{p}_s10'
        plot_density([StressModelData, StressModelLogN], filename, [f_data, f_lognormal], title=f"p={p}", save=True)
        plot_inv(filename, StressModelData, StressModelLogN, title=f"p={p}", save=True)

    # -------------------- Test Mean and Variance Optimisation -------------------- #
    mean_P_data, std_P_data = StressModelData.get_mean_std_baseline()
    mean_P_logn, std_P_logn = StressModelLogN.get_mean_std_baseline()

    _, _, _, _ = StressModelData.optimise_mean_std(mean_P_data, 1.2 * std_P_data)
    _, _, _, _ = StressModelLogN.optimise_mean_std(mean_P_logn, 1.2 * std_P_logn)

    filename = 'Plots/data_lognormal_M_S_20'
    plot_density([StressModelData, StressModelLogN], filename, [f_data, f_lognormal], save=True)
    plot_inv(filename, StressModelData, StressModelLogN, save=True)

    # -------------------- Test Utility and risk measure -------------------- #
    hara = lambda a, b, eta, x: (1 - eta) / eta * (a * x / (1 - eta) + b) ** eta

    b = lambda eta: 5 * (eta / (1 - eta)) ** (1 / eta)

    # Set gammas
    alpha = [0.8, 0.95]
    gammas = [lambda u: (u > alpha[0]) / (1 - alpha[0]), lambda u: (u > alpha[1]) / (1 - alpha[1])]

    StressModelData.set_gamma(gammas)
    StressModelLogN.set_gamma(gammas)

    RM_P_data = StressModelData.get_risk_measure_baseline()
    Utility_P_data = StressModelData.get_hara_utility(1, b(0.2), 0.2, StressModelData.u, StressModelData.F_inv)
    RM_P_logn = StressModelLogN.get_risk_measure_baseline()
    Utility_P_logn = StressModelLogN.get_hara_utility(1, b(0.2), 0.2, StressModelLogN.u, StressModelLogN.F_inv)

    utility_stresses = [0]
    rm_stresses = [-10, 10]

    for utility_stress in utility_stresses:
        _, _, _, _ = StressModelData.optimise_HARA(1, b(0.2), 0.2, Utility_P_data * (1 + utility_stress / 100),
                                                 RM_P_data * np.array([1 + rm_stresses[0] / 100, 1 + rm_stresses[1] / 100]))

        _, _, _, _ = StressModelLogN.optimise_HARA(1, b(0.2), 0.2, Utility_P_logn * (1 + utility_stress / 100),
                                                 RM_P_logn * np.array([1 + rm_stresses[0] / 100, 1 + rm_stresses[1] / 100]))

        filename = f'Plots/data_lognormal_utility_{utility_stress}_ES_{rm_stresses[0]}_{rm_stresses[1]}'

        plot_density([StressModelData, StressModelLogN], filename, [f_data, f_lognormal], save=True)
        plot_inv(filename, StressModelData, StressModelLogN, save=True)
