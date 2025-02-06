import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode45
from sklearn.cluster import KMeans


def plot_exemplary_trajectories(res_detail, props):
    # Convert to numpy array
    res_detail = np.array(res_detail)

    # Find unique solutions
    uniq_solutions = np.unique(res_detail[:, 2])
    n = len(uniq_solutions)

    # Extract initial conditions
    ICs = np.zeros((n, res_detail.shape[1]))
    for i in range(n):
        idx = np.where(res_detail[:, 2] == uniq_solutions[i])[0][0]
        ICs[i, :] = res_detail[idx, 0]
        print(f'initial condition for solution {i}: {res_detail[idx, 0]}')

    # Plotting
    plt.figure()
    ode_fun = props['model']['odeFun']
    tspan = props['ti']['tSpan']
    options = props['ti']['options']
    params = props['model']['odeParams']

    for i in range(n):
        T, Y = ode45(lambda t, y: ode_fun(t, y, *params),
                     tspan, ICs[i, :], options)
        plt.plot(T, Y[:, 0], label=f'solution {uniq_solutions[i]}')

    plt.xlabel('time')
    plt.ylabel('state 1')
    plt.title('exemplary trajectories')
    plt.legend()

    plt.savefig(f"{props['subCasePath']}/fig_sampleSolutions.png")
    plt.savefig(f"{props['subCasePath']}/fig_sampleSolutions.fig")
    plt.show()


def plot_bs_statespace(props, res_detail, idx_state1, idx_state2):
    Y0 = np.array([item[0] for item in res_detail])
    L = np.array([item[2] for item in res_detail])

    plt.figure()
    scatter = plt.scatter(Y0[:, idx_state1], Y0[:, idx_state2], c=L)
    plt.xlabel(f'state {idx_state1}', fontsize=12)
    plt.ylabel(f'state {idx_state2}', fontsize=12)
    plt.colorbar(scatter)

    plt.savefig(f"{props['subCasePath']}/fig_statespace.png")
    plt.savefig(f"{props['subCasePath']}/fig_statespace.fig")
    plt.show()


def plot_bs_parameter_study(props, tab, plot_error=False):
    num_solutions = (tab.shape[1] - 1) // 2
    nams = tab.columns[1:props['templ']['k'] + 2]

    plt.figure()
    p = []
    if plot_error:
        for i in range(num_solutions):
            plt.errorbar(tab.iloc[:, 0], tab.iloc[:, i + 1],
                         yerr=tab.iloc[:, i + 1 + num_solutions], label=f'class {i}')
    else:
        for i in range(num_solutions):
            plt.plot(tab.iloc[:, 0], tab.iloc[:, i + 1],
                     label=nams[i][3:], marker='.')

    plt.legend()
    plt.xlabel(f"model parameter {props['ap_study']['ap_name']}", fontsize=12)
    plt.ylabel('$\\mathcal{S}_{\\mathcal{B}}$', fontsize=12)

    plt.savefig(f"{props['subCasePath']}/fig_bs_study.png")
    plt.savefig(f"{props['subCasePath']}/fig_bs_study.fig")
    plt.show()


def plot_bs_hyperparameter_study(props, tab, plot_error=True):
    num_solutions = ((tab.shape[1] - 1) // 2) - 1
    class_names = props['templ']['label'] + ['NaN']

    plt.figure()
    p = []
    if plot_error:
        for i in range(num_solutions):
            plt.errorbar(tab.iloc[:, 0], tab.iloc[:, i + 1],
                         yerr=tab.iloc[:, i + 2 + num_solutions], label=class_names[i])
    else:
        for i in range(num_solutions):
            plt.plot(tab.iloc[:, 0], tab.iloc[:, i + 1],
                     label=class_names[i], marker='.', markersize=10)

    plt.legend()
    plt.xlabel(f"hyperparameter {props['ap_study']['ap_name']}", fontsize=12)
    plt.ylabel('basin stability', fontsize=12)

    plt.savefig(f"{props['subCasePath']}/fig_hyperparameter_study.png")
    plt.savefig(f"{props['subCasePath']}/fig_hyperparameter_study.fig")
    plt.show()


def plot_bs_featurespace(props, res_detail):
    X = np.array([item[1] for item in res_detail])
    L = np.array([item[2] for item in res_detail])

    plt.figure()
    if props['clust']['numFeatures'] == 1:
        for i in range(props['templ']['k']):
            plt.plot(props['templ']['features'][i][0], 'x',
                     markersize=6, linewidth=2, label='class templates')
    elif props['clust']['numFeatures'] == 2:
        plt.figure()
        for i in range(props['templ']['k']):
            plt.plot(props['templ']['features'][i][0], props['templ']['features']
                     [i][1], 'x', markersize=6, linewidth=2, label='class templates')
        scatter = plt.scatter(X[:, 0], X[:, 1], c=L)
        plt.legend()
        plt.xlabel('feature $X_1$', fontsize=12)
        plt.ylabel('feature $X_2$', fontsize=12)

    plt.title('feature space', fontsize=12)

    plt.savefig(f"{props['subCasePath']}/fig_featurespace.png")
    plt.savefig(f"{props['subCasePath']}/fig_featurespace.fig")
    plt.show()


def plot_bs_bargraph(props, res_tab, flag_plotErrorbar=False):
    plt.figure()
    b = plt.bar(range(len(res_tab)), res_tab['basinStability'], color=[.8, .8, .8], edgecolor=[
                0, 0, 0], linewidth=1.0)

    for i, rect in enumerate(b):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height,
                 f'{height:.3f}', ha='center', va='bottom')

    if flag_plotErrorbar:
        er = plt.errorbar(range(len(res_tab)), res_tab['basinStability'], yerr=res_tab['standardError'],
                          fmt='none', ecolor='black', capsize=5, label='standard error')
        plt.legend()

    plt.xticks(range(len(res_tab)), res_tab['label'])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('basin stability')
    plt.xlabel('solution label')

    plt.savefig(f"{props['subCasePath']}/fig_basinstability.png")
    plt.savefig(f"{props['subCasePath']}/fig_basinstability.fig")
    plt.show()


def plot_bif_diagram(props, res_detail, dof):
    n_dofs = len(dof)
    n_clusts = props['templ']['k']

    flag_par_var = 'ap_study' in props and props['ap_study']['mode'] == 'model_parameter'
    num_par_var = len(res_detail) if flag_par_var else 1

    amplitudes = np.full((n_clusts, n_dofs, num_par_var), np.nan)
    errs = np.full((n_clusts, n_dofs, num_par_var), np.nan)

    for idx_p in range(num_par_var):
        amplitudes[:, :, idx_p], errs[:, :, idx_p] = get_amplitudes(
            res_detail[idx_p] if flag_par_var else res_detail, dof, n_clusts)

    fig, axes = plt.subplots(1, n_dofs, sharey=True)
    for idx_d, ax in enumerate(axes):
        for idx_c in range(n_clusts):
            if flag_par_var:
                ax.plot(props['ap_study']['ap_values'],
                        amplitudes[idx_c, idx_d, :], 'k.')
            else:
                ax.plot(1, amplitudes[idx_c, idx_d, :], 'k.')
        ax.set_ylabel(f'amplitude state {dof[idx_d]}')

    plt.show()


def get_amplitudes(data_cell, dof, n_clust):
    n_dofs = len(dof)
    temp = np.array([item[4] for item in data_cell])[:, dof]

    kmeans = KMeans(n_clusters=n_clust)
    idx = kmeans.fit_predict(temp)
    amps = kmeans.cluster_centers_

    diffs = np.array([np.mean(np.abs(temp[idx == i] - amps[i]), axis=0)
                     for i in range(n_clust)])

    return amps, diffs
