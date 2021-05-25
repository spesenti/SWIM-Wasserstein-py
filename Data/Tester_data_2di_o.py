# 2dimensional input variables X and an output Y

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

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


def get_bivariate_KDE(data):
    # bivariate KDE for x1, x2
    h_y = 1.06 * np.std(data["y"]) * (len(data["y"])) ** (-1 / 5) / 2  # Silverman's rule, divide by 2 added
    h_x = 1.06 * np.std(data["x"], axis=0) * (len(data["y"])) ** (-1 / 5)

    f = lambda y: np.sum(norm.pdf((y.reshape(-1, 1) - data["y"].reshape(1, -1)) / h_y) / h_y / len(data["y"]),
                         axis=1).reshape(-1)
    F = lambda y: np.sum(norm.cdf((y.reshape(-1, 1) - data["y"].reshape(1, -1)) / h_y) / len(data["y"]), axis=1).reshape(-1)

    return h_y, h_x, f, F


def f_x(x1, x2, h_x, x_data):
    f = np.zeros(x1.shape)

    for i in range(len(x_data)):
        f += norm.pdf(x1, x_data[i, 0], h_x[0]) * norm.pdf(x2, x_data[i, 1], h_x[1])

    f /= len(x_data)

    return f


def gs_x(x1, x2, h_x, w, x_data):
    gs = np.zeros(x1.shape)

    for i in range(len(x_data)):
        gs += w[i] * norm.pdf(x1, x_data[i, 0], h_x[0]) * norm.pdf(x2, x_data[i, 1], h_x[1])

    gs /= np.sum(w)

    return gs


def generate_weights(data, StressModel):
    # Get smoother results, dependent on KDE approximation
    y_gd = np.linspace(StressModel.Gs_inv[3], StressModel.Gs_inv[-3], 500)

    print("computing gs at grid points...")
    gs = StressModel.gs(y_gd)
    gs /= integrate(gs, y_gd)

    print("computing f at grid points...")
    f = StressModel.f(y_gd)
    f /= integrate(f, y_gd)

    dQ_dP = gs / f

    print("E[dQ/dP]", integrate(dQ_dP * f, y_gd))

    print("computing weights...")
    w = np.zeros(len(data["y"]))
    for i in range(len(data["y"])):
        w[i] = integrate(norm.pdf((y_gd - data["y"][i]) / h_y) / h_y * dQ_dP, y_gd)
        # print( i, " of ", len(data["y"]))

    return w


def plot_dist(StressModel, filename, f, F, data, x1, x2, h_x, type="", title="", save=True):
    y_P = np.linspace(StressModel.F_inv[5], StressModel.F_inv[-5], 1000)
    y_Q = np.linspace(StressModel.Gs_inv[3], StressModel.Gs_inv[-3], 1000)

    _, gs, Gs = StressModel.distribution(StressModel.u, StressModel.Gs_inv, y_Q)

    fig = plt.figure(figsize=(5, 4))
    plt.plot(y_Q, gs, color='r', label='$g^*_Y$')
    plt.plot(y_P, f(y_P), '--', color='b', label='$f_Y$')
    plt.ylim(bottom=0)
    plt.xlim(1, 8)
    plt.title(title)
    plt.show()

    if save:
        fig.savefig(filename + '_density.pdf', format='pdf')

    fig = plt.figure(figsize=(5, 4))
    plt.plot(y_Q, Gs, color='r', label='$G^*_Y$')
    plt.plot(y_P, F(y_P), '--', color='b', label='$F_Y$')
    plt.xlim(1, 8)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=14)
    plt.title(title)
    plt.show()

    if save:
        fig.savefig(filename + '_CDF.pdf', format='pdf')

    idx = np.where(np.diff(StressModel.Gs_inv) < 1e-8)[0][0]

    fig = plt.figure(figsize=(4, 4))

    dQ_dP = gs / f(y_Q)
    if type == "ES":
        dQ_dP[:idx] = 1
    plt.plot(y_Q, dQ_dP)

    plt.ylim(0, 10)
    plt.xlim(3.8, 10)
    plt.xlabel(r"$y$", fontsize=18)
    plt.ylabel(r"$\frac{g^{*}_Y}{f_Y}$", fontsize=22)
    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig(filename + '_RN.pdf', format='pdf')

    # Get index for x- and y-values
    idx = (data["y"] >= y_Q[0]) & (data["y"] <= y_Q[-1])
    data_y_idx = data["y"][idx]
    data_x_idx = data["x"][idx]

    # Get g*/f
    w = StressModel.gs(data_y_idx) / StressModel.f(data_y_idx)

    f_x_plt = f_x(x1, x2, h_x, data_x_idx)
    gs_x_plt = gs_x(x1, x2, h_x, w, data_x_idx)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.contour(x1, x2, f_x_plt)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18)

    plt.subplot(1, 2, 2)
    plt.contour(x1, x2, gs_x_plt)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18)

    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(4, 4))
    f_x_plt_ctr = plt.contour(x1, x2, f_x_plt, alpha=0.8)
    plt.scatter(data_x_idx[:, 0], data_x_idx[:, 1], color='r', s=0.1)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18)
    plt.xlim(-1, 3)
    plt.ylim(-1, 3)
    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig(filename + '_data_x_P.pdf',format='pdf')

    fig = plt.figure(figsize=(4, 4))
    f_x_plt_ctr = plt.contour(x1, x2, f_x_plt, alpha=0.4)
    gs_x_plt_ctr = plt.contour(x1, x2, gs_x_plt, levels=f_x_plt_ctr.levels)
    plt.scatter(data_x_idx[:, 0], data_x_idx[:, 1], color='r', s=0.1)
    plt.xlim(-1, 3)
    plt.ylim(-1, 3)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18)
    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig(filename + '_data_x_Q.pdf',format='pdf')

    return


def plot_joint(x, filename, save=True):
    g = sns.jointplot(x=x[:, 0], y=x[:, 1], s=2)
    g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
    plt.xlim(-1, 4)
    plt.ylim(-1, 4)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18)
    plt.tight_layout()

    plt.show()

    if save:
        plt.savefig(filename + "_data_x.pdf", type="pdf")

    return


def plot_contour(x1, x2, f, filename, save=True):
    plt.figure(figsize=(4, 4))
    plt.contour(x1, x2, f_x(x1, x2, h_x, data["x"]))
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18)
    plt.show()

    if save:
        plt.savefig(filename + "_contour.pdf", type="pdf")

    return


# -------------------- Basic Metrics -------------------- #
def metrics(data, w):
    mean_y_P = np.mean(data["y"], axis=0)
    std_y_P = np.std(data["y"], axis=0)

    mean_x_P = np.mean(data["x"], axis=0)
    cov_x_P = np.cov(data["x"].T)
    corr_x_P = cov_x_P[0, 1] / np.sqrt(cov_x_P[0, 0] * cov_x_P[1, 1])

    P = {"mean_x": mean_x_P, "cov_x": cov_x_P, "corr_x": corr_x_P, "mean_y": mean_y_P, "std_y": std_y_P}

    mean_y_Q = np.sum(data["y"] * w, axis=0) / np.sum(w)
    std_y_Q = np.sqrt(np.sum((data["y"] - mean_y_Q) ** 2 * w, axis=0) / np.sum(w))

    mean_x_Q = np.sum(data["x"] * np.tile(w.reshape(-1, 1), (1, 2)), axis=0) / np.sum(w)
    cov_x_Q = np.zeros((2, 2))
    cov_x_Q[0, 0] = np.sum((data["x"][:, 0] - mean_x_Q[0]) ** 2 * w, axis=0) / np.sum(w)
    cov_x_Q[0, 1] = np.sum((data["x"][:, 0] - mean_x_Q[0]) * (data["x"][:, 1] - mean_x_Q[1]) * w, axis=0) / np.sum(w)
    cov_x_Q[1, 1] = np.sum((data["x"][:, 1] - mean_x_Q[1]) ** 2 * w, axis=0) / np.sum(w)
    cov_x_Q[1, 0] = cov_x_Q[0, 1]

    corr_x_Q = cov_x_Q[0, 1] / np.sqrt(cov_x_Q[0, 0] * cov_x_Q[1, 1])

    Q = {"mean_x": mean_x_Q, "cov_x": cov_x_Q, "corr_x": corr_x_Q, "mean_y": mean_y_Q, "std_y": std_y_Q}

    print("P", P)
    print("Q", Q)
    print("mean_x", Q["mean_x"] / P["mean_x"] - 1)
    print("std_x", np.sqrt(np.diag(Q["cov_x"]) / np.diag(P["cov_x"])) - 1)
    print("corr_x", Q["corr_x"] / P["corr_x"] - 1)
    print("std_y", Q["std_y"] / P["std_y"] - 1)

    return P, Q


if __name__ == "__main__":
    # -------------------- Generate data -------------------- #
    np.random.seed(0)
    Nsims = 1000
    p_mix = 0.5
    H = (np.random.uniform(size=Nsims) < p_mix)

    params = {"H1": {"mu": np.array([0.5, 1]), "cov": np.array([[0.2, 0.1], [0.1, 0.4]])},
              "H2": {"mu": np.array([1, 0.5]), "cov": np.array([[0.6, -0.3], [-0.3, 0.3]])}}

    n1 = np.random.multivariate_normal(mean=params["H1"]["mu"], cov=params["H1"]["cov"], size=Nsims)
    n2 = np.random.multivariate_normal(mean=params["H2"]["mu"], cov=params["H2"]["cov"], size=Nsims)

    x = np.tile((H == 0).reshape(-1, 1), (1, 2)) * n1 + np.tile((H == 1).reshape(-1, 1), (1, 2)) * n2

    filename = "Plots/Data-Generation/"
    plot_joint(x, filename, save=False)

    # Define the data and get the bandwidths, density and CDF
    data = {"y": 2 + np.exp(x[:, 0] + x[:, 1] - 3).reshape(-1), "x": x}
    h_y, h_x, f, F = get_bivariate_KDE(data)

    x1, x2 = np.meshgrid(np.linspace(-1, 4, 100), np.linspace(-1, 4, 100))
    plot_contour(x1, x2, f_x(x1, x2, h_x, data["x"]), filename, save=False)

    # -------------------- ES_0 and ES_0.95 -------------------- #
    alpha = [0, 0.95]
    u = create_u_grid([0.005, 0.95])

    gamma = [lambda u: (u > alpha[0]) / (1 - alpha[0]), lambda u: (u > alpha[1]) / (1 - alpha[1])]

    # for i in range(len(gamma)):
    #     plt.plot(u, gamma[i](u))
    # plt.ylabel(r'$\gamma(u)$')
    # plt.xlabel('u')
    # plt.show()

    # -------------------- Generate the model -------------------- #
    StressModel = W_Stress(data, u)

    # -------------------- Optimize ES risk measure -------------------- #
    StressModel.set_gamma(gamma)
    RM_P = StressModel.get_risk_measure_baseline()
    lam, WD, RM_Q, fig = StressModel.optimise_rm(RM_P * np.array([1.05, 1.05]))

    filename = 'Plots/2D/ES/data_ES_0_u5_95_u5'
    # fig.savefig(filename + '_inv.pdf',format='pdf')

    plot_dist(StressModel, filename, f, F, data, x1, x2, h_x, "ES", save=False)

    w = generate_weights(data, StressModel)
    P, Q = metrics(data, w)

    # -------------------- Optimize ES risk measure -------------------- #
    lam, WD, RM_Q, fig = StressModel.optimise_rm(RM_P * np.array([0.95, 1.05]))

    filename = 'Plots/2D/ES/data_ES_0_d5_95_u5'
    # fig.savefig(filename + '_inv.pdf',format='pdf')

    plot_dist(StressModel, filename, f, F, data, x1, x2, h_x, "ES", save=False)

    w = generate_weights(data, StressModel)

    P, Q = metrics(data, w)

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

        filename = f'Plots/2D/alpha-beta/data_alpha_{alpha}_beta_{beta}_p_{p}_10'
        fig.savefig(filename + '_inv.pdf', format='pdf')

        plot_dist(StressModel, filename, f, F, data, x1, x2, h_x, "ES", title=f"p={p}")

    # -------------------- Test Mean and Variance Optimisation -------------------- #
    mean_P, std_P = StressModel.get_mean_std_baseline()

    mean_stress = 20
    std_stress = -20
    lam, WD, mv_Q, fig = StressModel.optimise_mean_std((1 + mean_stress/100) * mean_P, (1 + std_stress/100) * std_P)

    filename = f'Plots/2D/mean-std/data_M{mean_stress}_S{std_stress}'
    fig.savefig(filename + '_inv.pdf',format='pdf')

    plot_dist(StressModel, filename, f, F, data, x1, x2, h_x, 'mean-std', save=True)

    # -------------------- Test mean-variance + ES measure -------------------- #
    alpha = 0.95
    gamma_ES = [lambda u: (u >= alpha) / (1 - alpha)]

    StressModel.set_gamma(gamma_ES)

    RM_P = StressModel.get_risk_measure_baseline()
    mean_P, std_P = StressModel.get_mean_std_baseline()

    rm_stresses = [10, 12, 15]
    mean_stresses = [0, 0, 0]
    std_stresses = [0, 0, 0]

    # stresses = [-10, 0, 10]
    # for stress in itertools.product(stresses, repeat=3):

    for i in range(len(rm_stresses)):
        rm_stress = rm_stresses[i]
        mean_stress = mean_stresses[i]
        std_stress = std_stresses[i]
        _, _, RM_Q, _, fig = StressModel.optimise_rm_mean_std(np.array([1 + rm_stress / 100]) * RM_P,
                                                              (1 + mean_stress / 100) * mean_P,
                                                              (1 + std_stress / 100) * std_P)

        filename = f'Plots/2D/ES-mean-std/data_alpha_{alpha}_ES_{rm_stress}_M_{mean_stress}_S_{std_stress}'
        fig.savefig(filename + '_inv.pdf', format='pdf')

        plot_dist(StressModel, filename, f, F, data, x1, x2, h_x, "rm-mean-std", save=True)

    # -------------------- Test Utility and risk measure -------------------- #
    # ******** NOT Converging ********
    hara = lambda a, b, eta, x: (1 - eta) / eta * (a * x / (1 - eta) + b) ** eta

    b = lambda eta: 5 * (eta / (1 - eta)) ** (1 / eta)
    y = np.linspace(1e-20, 30, 1000)
    plt.plot(y, hara(1, b(0.2), 0.2, y))

    # Set gammas
    alpha = [0.8, 0.95]
    gammas = [lambda u: (u > alpha[0]) / (1 - alpha[0]), lambda u: (u > alpha[1]) / (1 - alpha[1])]
    StressModel.set_gamma(gammas)

    RM_P = StressModel.get_risk_measure_baseline()
    Utility_P = StressModel.get_hara_utility(1, b(0.2), 0.2, StressModel.u, StressModel.F_inv)

    utility_stresses = [0, 0.5, 1]
    rm_stresses = [-3, 3]

    for utility_stress in utility_stresses:
        _, _, _, fig = StressModel.optimise_HARA(1, b(0.2), 0.2, Utility_P * (1 + utility_stress / 100),
                                                 RM_P * np.array([1 + rm_stresses[0] / 100, 1 + rm_stresses[1] / 100]))

        filename = f'Plots/2D/HARA-ES/data_utility_{utility_stress}_ES_{rm_stresses[0]}_{rm_stresses[1]}'
        fig.savefig(filename + '_inv.pdf', format='pdf')

        plot_dist(StressModel, filename, f, F, data, x1, x2, h_x, "Utility")

