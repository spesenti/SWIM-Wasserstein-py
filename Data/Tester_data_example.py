import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma
import seaborn as sns
import itertools
import pandas as pd
import time

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


def get_KDE(data):
    # bivariate KDE for x1, x2
    h_y = 1.06 * np.std(data["y"]) * (len(data["y"])) ** (-1 / 5)  # Silverman's rule
    h_x = 1.06 * np.std(data["x"], axis=0) * (len(data["y"])) ** (-1 / 5)

    f = lambda y: np.sum(norm.pdf((y.reshape(-1, 1) - data["y"].reshape(1, -1)) / h_y) / h_y / len(data["y"]),
                         axis=1).reshape(-1)
    F = lambda y: np.sum(norm.cdf((y.reshape(-1, 1) - data["y"].reshape(1, -1)) / h_y) / len(data["y"]), axis=1).reshape(-1)

    return h_y, h_x, f, F


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


def generate_data(Nsims, plot=True):

    # Get z1, ... , z10 where zi = (zi(1), zi(2)) and zi(1), zi(2) ~ U(0,1)
    z1 = np.random.uniform(size=10)
    z2 = np.random.uniform(size=10)
    z = list(zip(z1, z2))

    # Get theta in (0, 0.4, 5) with probabilities (0.05, 0.6, 0.35)
    x = np.random.uniform(0, 1, size=Nsims)
    thetas = np.zeros(Nsims)

    thetas[np.where(x <= 0.05)] = 0
    thetas[np.where((0.05 < x) & (x <= 0.65))] = 0.4
    thetas[np.where(x > 0.65)] = 5

    # Generate realizations of (L1, ..., L10) using Gaussian copulas
    x = np.zeros((Nsims, 10))
    theta_labels = {5:[], 0.4:[], 0: []}
    for i in range(Nsims):
        theta = thetas[i]
        # Get the correlation
        # rho_ij = exp{-theta_i * ||zi - zj||} where ||.|| denotes the Euclidean distance
        Omega = np.zeros((10, 10))
        for j in range(10):
            for k in range(10):
                Omega[j, k] = np.exp(-theta * np.linalg.norm(np.array(z[j]) - np.array(z[k])))

        # Create samples from a correlated multivariate normal
        x0 = np.random.multivariate_normal(mean=np.zeros(10), cov=Omega)
        x[i, :] = x0
        theta_labels[theta].append(i)

    # Get uniform marginals
    u = norm.cdf(x)

    # Marginal distributions Li ~ Gamma(5, 0.2i) with mean=25
    L = np.zeros((Nsims, 10))
    x_axis = np.linspace(25, 50, 200)
    means = np.zeros(10)
    for i in range(10):
        L_i = gamma.ppf(u[:, i], a=5, loc=25, scale=0.2*(i+1))
        L[:, i] = L_i

        means[i] = np.mean(L_i)

        # Gamma distribution plot
        if plot:
            y_i = gamma.pdf(x_axis, a=5, loc=25, scale=0.2*(i+1))
            plt.plot(x_axis, y_i, label=f"scale={0.2*(i+1)}")
    if plot:
        plt.legend()
        plt.savefig('Plots/ex/data_marginal_dist.pdf', format='pdf')
        plt.show()

    max_mean = np.max(means)
    min_mean = np.min(means)
    marker_sizes = (means - min_mean)/(max_mean - min_mean)*150

    # Location plot
    if plot:
        plt.scatter(z1, z2, marker='o', color='black', s=marker_sizes)
        plt.savefig('Plots/ex/data_location_by_mean.pdf', format='pdf')
        plt.show()

    # Define the data and get the bandwidths, density and CDF
    data = {"y": np.sum(L, axis=1), "x": L}

    return data, theta_labels


if __name__ == "__main__":
    # -------------------- Generate data -------------------- #
    np.random.seed(1)
    data, theta_labels = generate_data(1000, plot=False)

    h_y, h_x, f, F = get_KDE(data)

    # -------------------- Visualize data -------------------- #
    colors = {0: 'black', 0.4: 'red', 5:'green'}
    filename = 'Plots/ex/theta/'

    # Pair-wise scatterplot
    df_L = pd.DataFrame(data['x'])
    df_L['theta'] = np.nan
    for label in [0, 0.4, 5]:
        df_L.iloc[theta_labels[label], 10] = str(label)
    pp = sns.pairplot(df_L, hue='theta', plot_kws={"s": 3})
    plt.savefig(filename + 'pairwise_plt.pdf', format='pdf')
    plt.show()

    # Individual plots
    for pair in list(itertools.combinations(range(10), 2)):
        for theta in theta_labels.keys():
            row_idx = theta_labels[theta]
            plt.scatter(data['x'][row_idx, pair[0]], data['x'][row_idx, pair[1]], color=colors[theta],
                        s=8, label=f"$\Theta={theta}$")
        plt.xlabel(f"$L_{pair[0]+1}$")
        plt.ylabel(f"$L_{pair[1]+1}$")
        plt.legend()
        plt.savefig(filename + f'individual_{pair[0]+1}_{pair[1]+1}.pdf', format='pdf')
        plt.show()

    plt.scatter(data['x'][:, 9], data['y'])
    plt.xlabel("$L_{10}$")
    plt.ylabel(f"$Y$")
    plt.legend()
    plt.show()

    # -------------------- Generate the model -------------------- #
    u = create_u_grid([0.005, 0.95])
    StressModel = W_Stress(data, u, [200, 450])
    sensitivity_measures = []
    alt_sensitivity_measures = []
    delta_P = []
    delta_Q = []
    labels = []
    colors = []

    # StressModel.plot_f_F()

    # -------------------- Optimize ES risk measure -------------------- #
    alpha = [0.8, 0.95]
    gamma = [lambda u: (u > alpha[0]) / (1 - alpha[0]), lambda u: (u > alpha[1]) / (1 - alpha[1])]

    StressModel.set_gamma(gamma)
    RM_P = StressModel.get_risk_measure_baseline()

    ES_stresses = [[0, 1], [5, 5]]
    colors = colors + ['orange', 'violet']
    for stress in ES_stresses:
        lam, WD, RM_Q, fig = StressModel.optimise_rm(RM_P * np.array([1 + stress[0]/100, 1 + stress[1]/100]))

        filename = f'Plots/ex/ES/data_ES_80_{stress[0]}_95_{stress[1]}'
        fig.savefig(filename + '_inv.pdf',format='pdf')

        StressModel.plot_dist(filename, type="ES", save=True)
        print(StressModel.reverse_sensitivity_measure(lambda x: x, StressModel.data['x']))
        sensitivity_measures.append(StressModel.reverse_sensitivity_measure(lambda x: x, StressModel.data['x']))
        labels.append(f'$ES_{{80}}$ {stress[0]}% | $ES_{{95}}$ {stress[1]}%')
    StressModel.plot_sensitivities(sensitivity_measures,
    f'Plots/ex/ES/data_ES_80_{stress[0]}_95_{stress[1]}_sensitivity.pdf', colors, labels, title='s(x)=x', save=True)

    # -------------------- Optimize alpha-beta risk measure -------------------- #
    p_list = [0.25, 0.5, 0.75]
    alpha = 0.9
    beta = 0.1
    colors = colors + ['black', 'grey', 'indigo']

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

        filename = f'Plots/ex/alpha-beta/data_alpha_{alpha}_beta_{beta}_p_{p}_10'
        fig.savefig(filename + '_inv.pdf', format='pdf')

        StressModel.plot_dist(filename, type="ES", title=f"p={p}", save=True)
        sensitivity_measures.append(StressModel.reverse_sensitivity_measure(lambda x: x, StressModel.data['x']))
        labels.append(f'p={p}')
    StressModel.plot_sensitivities(sensitivity_measures,
    f'Plots/ex/alpha-beta/data_alpha_{alpha}_beta_{beta}_10_sensitivity.pdf', colors, labels, title='s(x)=x', save=True)

    # -------------------- Test Mean and Variance Optimisation -------------------- #
    mean_P, std_P = StressModel.get_mean_std_baseline()

    mean_stress = [10]
    std_stress = [2]

    colors = colors + ['r', 'g', 'b']
    for i in range(len(mean_stress)):
        lam, WD, mv_Q, fig = StressModel.optimise_mean_std((1 + mean_stress[i]/100) * mean_P, (1 + std_stress[i]/100) * std_P)

        filename = f'Plots/ex/mean-std/data_M{mean_stress[i]}_S{std_stress[i]}'
        # fig.savefig(filename + '_inv.pdf',format='pdf')

        StressModel.plot_dist(filename, type='mean-std', save=False)

        sensitivity_measures.append(StressModel.reverse_sensitivity_measure(lambda x: x, StressModel.data['x']))
        results = StressModel.alternate_sensitivity_measure()
        alt_sensitivity_measures.append(results[0])
        delta_P.append(results[1])
        delta_Q.append(results[2])
        labels.append(f'mean {mean_stress[i]}% | std {std_stress[i]}%')

    StressModel.plot_sensitivities(sensitivity_measures, filename = f'Plots/ex/mean-std/data_M_S_sensitivity.pdf',
                                   labels=labels, colors=colors, title=r's(X)=X', save=False)
    StressModel.plot_sensitivities(alt_sensitivity_measures, filename = f'Plots/ex/mean-std/data_M_S_alt_sensitivity.pdf',
                                   labels=labels, colors=colors, title=r'$\frac{\delta_Q - \delta_P}{\delta_P}$', save=True)
    P_ranks = np.argsort(np.argsort(-delta_P[0])) + 1
    Q_ranks = np.argsort(np.argsort(-delta_Q[0])) + 1
    print(alt_sensitivity_measures, delta_P, P_ranks, delta_Q, Q_ranks)
    alt_sensitivity_df = pd.DataFrame({ 'Reverse Sensitivity Measure': list(sensitivity_measures[0]),
                                        'Alternate Sensitivity Measure': list(alt_sensitivity_measures[0]),
                                        'delta_P': list(delta_P[0]),
                                        'P_rank': list(P_ranks),
                                        'delta_Q': list(delta_Q[0]),
                                        'Q_rank': list(Q_ranks)
                                        }, index = [f'X{i + 1}' for i in range(len(alt_sensitivity_measures[0]))])
    alt_sensitivity_df.to_csv(f'Plots/ex/mean-std/data_M_S_sensitivity_stats.csv')

    # -------------------- Test mean-variance + ES measure -------------------- #
    alpha = 0.95
    gamma_ES = [lambda u: (u >= alpha) / (1 - alpha)]

    StressModel.set_gamma(gamma_ES)

    RM_P = StressModel.get_risk_measure_baseline()
    mean_P, std_P = StressModel.get_mean_std_baseline()

    rm_stresses = [10, 5, 0]
    mean_stresses = [10, -5, 0]
    std_stresses = [-10, 0, 20]
    colors = colors + ['r', 'g', 'b']

    for i in range(len(rm_stresses)):
        rm_stress = rm_stresses[i]
        mean_stress = mean_stresses[i]
        std_stress = std_stresses[i]
        _, _, RM_Q, _, fig = StressModel.optimise_rm_mean_std(np.array([1 + rm_stress / 100]) * RM_P,
                                                              (1 + mean_stress / 100) * mean_P,
                                                              (1 + std_stress / 100) * std_P)

        filename = f'Plots/ex/ES-mean-std/data_alpha_{alpha}_ES_{rm_stress}_M_{mean_stress}_S_{std_stress}'
        fig.savefig(filename + '_inv.pdf', format='pdf')

        sensitivity_measures.append(StressModel.reverse_sensitivity_measure(lambda x: x, StressModel.data['x']))
        labels.append(f'mean {mean_stress}% | std {std_stress}% | ES {rm_stress}%')

        StressModel.plot_dist(filename, type="rm-mean-std", save=True)

    StressModel.plot_sensitivities(sensitivity_measures, filename=f'Plots/ex/ES-mean-std/data_ES_M_S_sensitivity.pdf',
                                   labels=labels, colors=colors, title=r's(X)=X', save=True)

    # -------------------- Test Utility and risk measure -------------------- #
    # ******** NOT Converging ********
    hara = lambda a, b, eta, x: (1 - eta) / eta * (a * x / (1 - eta) + b) ** eta

    # Find concave utility, maybe try different parameters? send current params in slack, range of y (from simulation)

    b = lambda eta: 5 * (eta / (1 - eta)) ** (1 / eta)
    y = np.linspace(1e-20, 30, 1000)
    # plt.plot(y, hara(1, b(0.2), 0.2, y))
    # plt.show()

    # Set gammas
    alpha = [0.8, 0.95]
    gammas = [lambda u: (u > alpha[0]) / (1 - alpha[0]), lambda u: (u > alpha[1]) / (1 - alpha[1])]
    StressModel.set_gamma(gammas)

    RM_P = StressModel.get_risk_measure_baseline()
    a = 1
    eta = 0.5
    Utility_P = StressModel.get_hara_utility(a, b(eta), eta, StressModel.u, StressModel.F_inv)

    utility_stresses = [0, 1]
    rm_stresses = [[0, 1], [1, 3]]

    colors = colors + ['pink', 'purple', 'blue']
    for i in range(len(utility_stresses)):
        start_time = time.time()
        utility_stress = utility_stresses[i]
        rm_stress = rm_stresses[i]

        _, _, _, fig = StressModel.optimise_HARA(a, b(eta), eta, Utility_P * (1 + utility_stress / 100),
                                                 RM_P * np.array([1 + rm_stress[0] / 100, 1 + rm_stress[1] / 100]))

        print(f"HARA optimization for a={a}, b={b(eta)}, eta={eta}, util_stress={utility_stress} took {round(time.time() - start_time, 2)} seconds ")
        filename = f'Plots/ex/hara-es/data_utility_{utility_stress}_ES_{rm_stress[0]}_{rm_stress[1]}'
        # fig.savefig(filename + '_inv.pdf', format='pdf')

        StressModel.plot_dist(filename, type="Utility", save=False)

        # S = np.zeros(10)
        # for k in range(10):
        #     # Get s:R->R
        #     level = 0.95
        #     s = lambda x: x > gamma.ppf(level, a=5, loc=25, scale=0.2 * (k + 1))
        #     S[k] = StressModel.reverse_sensitivity_measure(s, StressModel.data['x'][:, k])
        # print(S)
        S = StressModel.reverse_sensitivity_measure(lambda x: x, StressModel.data['x'])
        sensitivity_measures.append(S)

        # results = StressModel.alternate_sensitivity_measure()
        # alt_sensitivity_measures.append(results[0])
        # delta_P.append(results[1])
        # delta_Q.append(results[2])
        #
        # P_ranks = np.argsort(np.argsort(-delta_P[i])) + 1
        # Q_ranks = np.argsort(np.argsort(-delta_Q[i])) + 1
        # print(alt_sensitivity_measures, delta_P, P_ranks, delta_Q, Q_ranks)
        # alt_sensitivity_df = pd.DataFrame({'Reverse Sensitivity Measure': list(sensitivity_measures[i]),
        #                                    'Alternate Sensitivity Measure': list(alt_sensitivity_measures[i]),
        #                                    'delta_P': list(delta_P[i]),
        #                                    'P_rank': list(P_ranks),
        #                                    'delta_Q': list(delta_Q[i]),
        #                                    'Q_rank': list(Q_ranks)
        #                                    }, index=[f'X{k + 1}' for k in range(len(alt_sensitivity_measures[i]))])
        # alt_sensitivity_df.to_csv(f'Plots/ex/hara-es/data_utility_{utility_stress}_ES_{rm_stress[0]}_{rm_stress[1]}_sensitivity_stats.csv')

        labels.append(f'utility {utility_stress}% | $ES_{{80}}$ {rm_stress[0]}% | $ES_{{95}}$ {rm_stress[1]}%')

    StressModel.plot_sensitivities(sensitivity_measures, labels=labels, colors=colors,
                                   # title=r's($X_i$) = I{$X_i > \breve{F_i}(0.95)$}', save=True,
                                   # filename=f'Plots/ex/hara-es/data_ES_utility_sensitivity_{level}.pdf')
                                   title=r's($X$) = X', save=False,
                                   filename=f'Plots/ex/hara-es/data_ES_utility_sensitivity.pdf')
    # StressModel.plot_sensitivities(alt_sensitivity_measures, filename = f'Plots/ex/hara-es/data_ES_utility_sensitivity_alt_sensitivity.pdf',
    #                                labels=labels, colors=colors, title=r'$\frac{\delta_Q - \delta_P}{\delta_P}$', save=True)
