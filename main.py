import numpy as np
import warnings
import logging
import nashpy as nash
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from numpy.random import default_rng

np.seterr(all="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.ERROR)

states = 50
actions = 3
horizon = 100
cb_all = [1e-3, 5e-3, 1e-2, 5e-2]
sigma_all = [0.2]
data_size_all = [224]
seed_num = 20
S, actions, N, H = 50, 3, 2, 100
A = actions ** N
xi_base = 2


def gen_ground_truth_zerosum(rng):
    P = rng.random((S, A, S))
    P /= P.sum(-1, keepdims=True)
    r_scalar = rng.uniform(0, 1, size=(S, A))
    R = np.stack([r_scalar, -r_scalar])
    return P, R


def perturb_tv(P, sigma, rng):
    noise = rng.uniform(-sigma, sigma, P.shape)
    P_tilde = np.clip(P + noise, 0, None)
    P_tilde /= P_tilde.sum(-1, keepdims=True)
    return P_tilde


def sample_dataset(P, R, size, sigma, rng):
    P_off = perturb_tv(P, sigma, rng)
    D = []
    for _ in range(size):
        s = rng.integers(S)
        a1 = rng.integers(actions)
        a2 = rng.integers(actions)
        a = a1 * actions + a2
        s_ = rng.choice(S, p=P_off[s, a])
        D.append((s, a, R[:, s, a], s_))
    return D


def estimate_mle(D):
    P_cnt = np.zeros((S, A, S))
    R_sum, R_cnt = np.zeros((N, S, A)), np.zeros((S, A))
    for s, a, r, s_ in D:
        P_cnt[s, a, s_] += 1
        R_sum[:, s, a] += r
        R_cnt[s, a] += 1
    P_hat = (P_cnt + 1e-6)
    P_hat /= P_hat.sum(-1, keepdims=True)
    R_hat = np.divide(R_sum, R_cnt[None, ...], where=R_cnt > 0, out=np.zeros_like(R_sum))
    return P_hat, R_hat


def sample_confidence_models_L1(
    P_hat: np.ndarray,
    traj_sa: list,
    rng: np.random.Generator,
    xi: float,
    K: int = 10,
):
    S, A, _ = P_hat.shape
    models = []

    idx_s = np.array([t[0] for t in traj_sa], dtype=int)
    idx_a = np.array([t[1] for t in traj_sa], dtype=int)

    for _ in range(K):
        U = rng.dirichlet(np.ones(S), size=(S, A)).reshape(S, A, S)
        diff = np.abs(P_hat[idx_s, idx_a] - U[idx_s, idx_a]).sum(-1)
        d = np.mean(diff**2)

        if d == 0.0:
            models.append(P_hat.copy())
            continue

        alpha = min(1.0, math.sqrt(max(xi, 1e-12) / max(d, 1e-12)))

        P_tmp = (1 - alpha) * P_hat + alpha * U

        P_tmp = np.clip(P_tmp, 0.0, None)
        P_tmp /= P_tmp.sum(-1, keepdims=True)

        models.append(P_tmp)

    return models, alpha


def solve_zero_sum_general(payoff):
    """payoff shape = (A, A). return (row_dist, col_dist)"""
    game = nash.Game(payoff, -payoff)
    try:
        strat_row, strat_col = game.lemke_howson(initial_dropped_label=0)
    except Exception:
        strat_row, strat_col = next(game.support_enumeration())
    return np.asarray(strat_row), np.asarray(strat_col)


def robust_Q(pi_pair, Pset, R, pess=True):
    pi1, pi2 = pi_pair
    V = np.zeros((H + 1, N, S))
    Q = np.zeros((H, N, S, A))
    for h in range(H - 1, -1, -1):
        Ph = (np.stack(Pset[h]).min(0) if pess else np.stack(Pset[h]).max(0))
        Ph /= Ph.sum(-1, keepdims=True)
        Q[h] = R + np.einsum("sat,nt->nsa", Ph, V[h + 1])
        joint = np.einsum("sa,sb->sab", pi1[h], pi2[h]).reshape(S, A)
        V[h] = np.einsum("sa,nsa->ns", joint, Q[h])
    return Q, V[0]


def mixed_gap(pi1, pi2, delta):
    joint = np.einsum("hsa,hsb->hsab", pi1, pi2).reshape(H, S, A)
    g = (joint * delta[:, 0]).sum(-1)
    return g.max()


def safe_equilibrium(M, actions, seed=None):
    game = nash.Game(M, -M)

    def _try_lehmer(game):
        try:
            p, q = game.lemke_howson(initial_dropped_label=0)
        except Exception:
            return None, None
        if np.any(np.isnan(p)) or np.any(np.isnan(q)):
            return None, None
        return np.asarray(p, float), np.asarray(q, float)

    p, q = _try_lehmer(game)
    if p is None:
        rng = np.random.default_rng(seed)
        eps = 1e-8
        noise = rng.uniform(-eps, eps, size=M.shape)
        p, q = _try_lehmer(nash.Game(M + noise, -(M + noise)))

    if p is None:
        try:
            p, q = next(game.support_enumeration())
        except StopIteration:
            p = q = None

    if p is None:
        p = np.ones(actions) / actions
        q = np.ones(actions) / actions

    if p.size < actions:
        p = np.pad(p, (0, actions - p.size))
    if q.size < actions:
        q = np.pad(q, (0, actions - q.size))
    return p, q


def one_step_best_response(pi1, pi2, Pset, R, eta=1):
    Q_pess, _ = robust_Q((pi1, pi2), Pset, R, True)
    Q_opt, _ = robust_Q((pi1, pi2), Pset, R, False)
    delta = Q_opt - Q_pess

    for h in range(H):
        for s in range(S):
            M = delta[h, 0, s].reshape(actions, actions)
            p, q = safe_equilibrium(M, actions, seed=(h * 13 + s))
            pi1[h, s] = (1 - eta) * pi1[h, s] + eta * p
            pi2[h, s] = (1 - eta) * pi2[h, s] + eta * q

    Q_pess, _ = robust_Q((pi1, pi2), Pset, R, True)
    Q_opt, _ = robust_Q((pi1, pi2), Pset, R, False)
    return pi1, pi2, mixed_gap(pi1, pi2, Q_opt - Q_pess)


def estimate_mle_pair(D_pair):
    A = actions**2
    P_cnt = np.zeros((S, A, S))
    R_sum = np.zeros((2, S, A))
    R_cnt = np.zeros((S, A))

    for s, a1, a2, r, s_next in D_pair:
        a = a1 * actions + a2
        P_cnt[s, a, s_next] += 1

        R_sum[0, s, a] += r
        R_sum[1, s, a] -= r
        R_cnt[s, a] += 1

    P_hat = (P_cnt + 1e-6)
    P_hat /= P_hat.sum(-1, keepdims=True)

    R_hat = np.divide(R_sum, R_cnt[None, ...], where=R_cnt > 0, out=np.zeros_like(R_sum))
    return P_hat, R_hat


def generate_transition():
    P = np.zeros((actions, actions, states, states))
    R = np.random.random((actions, actions, states))
    for a1 in range(actions):
        for a2 in range(actions):
            for s in range(states):
                P[a1, a2, s] = np.random.random(states)
                P[a1, a2, s] /= P[a1, a2, s].sum()
    return P, R


def perturb_transition_tv(P0, sigma):
    A1, A2, S, _ = P0.shape
    P_perturbed = np.zeros_like(P0)

    for a1 in range(A1):
        for a2 in range(A2):
            for s in range(S):
                p0 = P0[a1, a2, s]
                noise = np.random.uniform(-sigma, sigma, size=S)
                p_tilde = p0 + noise
                p_tilde = np.clip(p_tilde, 0, 1)
                if p_tilde.sum() > 0:
                    p_tilde /= p_tilde.sum()
                else:
                    noise = np.random.uniform(-sigma, sigma, size=S)
                    p_tilde = p0 + noise
                    p_tilde = np.clip(p_tilde, 0, 1)
                tv = 0.5 * np.sum(np.abs(p_tilde - p0))
                if tv > sigma:
                    p_tilde = (1 - sigma) * p0 + sigma * np.random.dirichlet(np.ones(S))
                P_perturbed[a1, a2, s] = p_tilde
    return P_perturbed


def generate_dataset(P, R, dataset_size, sigma):
    dataset = []
    P_perturbed = perturb_transition_tv(P, sigma)
    for _ in range(dataset_size):
        s = np.random.randint(0, states)
        a1 = np.random.randint(0, actions)
        a2 = np.random.randint(0, actions)
        s_new = np.random.choice(states, p=P_perturbed[a1, a2, s])
        r = R[a1, a2, s]
        dataset.append((s, a1, a2, r, s_new))
    return dataset


def estimate_dataset(dataset):
    P_hat = np.zeros((actions, actions, states, states))
    R_sum = np.zeros((actions, actions, states))
    R_count = np.zeros((actions, actions, states))

    for (s, a1, a2, r, s_new) in dataset:
        P_hat[a1, a2, s, s_new] += 1
        R_sum[a1, a2, s] += r
        R_count[a1, a2, s] += 1

    for a1 in range(actions):
        for a2 in range(actions):
            for s in range(states):
                total = P_hat[a1, a2, s].sum()
                if total > 0:
                    P_hat[a1, a2, s] /= total
                else:
                    P_hat[a1, a2, s] = np.ones(states) / states

    R_hat = np.zeros((actions, actions, states))
    for a1 in range(actions):
        for a2 in range(actions):
            for s in range(states):
                count = R_count[a1, a2, s]
                if count > 0:
                    R_hat[a1, a2, s] = R_sum[a1, a2, s] / count
                else:
                    R_hat[a1, a2, s] = 0.0
    return P_hat, R_hat, R_count


def generate_best(P, R):
    policy_max = np.zeros((horizon, states, actions))
    policy_min = np.zeros((horizon, states, actions))
    V = np.zeros((horizon + 1, states))

    for h in range(horizon - 1, -1, -1):
        for s in range(states):
            Q = np.zeros((actions, actions))
            for a1 in range(actions):
                for a2 in range(actions):
                    expected_value = P[a1, a2, s] @ V[h + 1]
                    Q[a1, a2] = R[a1, a2, s] + expected_value
            game = nash.Game(Q)

            try:
                eq = game.lemke_howson(0)
            except Exception:
                try:
                    eq = next(game.support_enumeration())
                except StopIteration:
                    eq = (np.ones(actions) / actions, np.ones(actions) / actions)

            policy_s_max, policy_s_min = eq
            V[h, s] = game[policy_s_max, policy_s_min][0]
            policy_max[h, s] = policy_s_max
            policy_min[h, s] = policy_s_min
    return V, policy_max, policy_min


def bonus(p_hat, z, n_sa, cb, h, H=horizon):
    mean = p_hat @ z
    var = p_hat @ (z**2) - mean**2
    var = np.where(np.abs(var) < 1e-8, 0, var)

    bernstein_1 = np.zeros_like(n_sa, dtype=np.float32)
    bernstein_2 = np.zeros_like(n_sa, dtype=np.float32)
    mask = n_sa > 0
    bernstein_1[mask] = np.sqrt(cb * var[mask] / n_sa[mask])
    bernstein_2[mask] = (2 * cb * H) / (n_sa[mask])
    bernstein = np.maximum(bernstein_1, bernstein_2)
    return np.minimum(bernstein, H)


def alpha_max_backup(V_next, P0, sigma):
    """
    Implements the max over alpha formulation of robust Bellman backup.

    Args:
        V_next: shape (S,), next-step estimated value vector
        P0: shape (S,), estimated transition probability (sparse distribution)
        sigma: float, robustness radius

    Returns:
        backup_value: float, robust value after optimization
    """
    v_min = np.min(V_next)
    v_max = np.max(V_next)
    out = np.zeros(states)
    for i in range(states):
        def objective(alpha):
            V_alpha = np.minimum(V_next, alpha)
            penalty = sigma * (alpha - np.min(V_alpha))
            return -((P0 @ V_alpha)[i] - penalty)

        res = minimize_scalar(objective, bounds=(v_min, v_max), method="bounded")
        out[i] = -res.fun
    return out


def alpha_min_backup(V_next, P0, sigma):
    """
    Implements the max over alpha formulation of robust Bellman backup.

    Args:
        V_next: shape (S,), next-step estimated value vector
        P0: shape (S,), estimated transition probability (sparse distribution)
        sigma: float, robustness radius

    Returns:
        backup_value: float, robust value after optimization
    """
    v_min = np.min(V_next)
    v_max = np.max(V_next)
    out = np.zeros(states)
    for i in range(states):
        def objective(alpha):
            V_alpha = np.maximum(V_next, alpha)
            penalty = sigma * (alpha - np.max(V_alpha))
            return (P0 @ V_alpha)[i] - penalty

        res = minimize_scalar(objective, bounds=(v_min, v_max), method="bounded")
        out[i] = res.fun
    return out


def generate_VI(P, R, sigma):
    V_min = np.zeros((horizon + 1, states))
    V_max = np.zeros((horizon + 1, states))
    Q_min = np.zeros((actions, actions, states))
    Q_max = np.zeros((actions, actions, states))
    Q_opt = np.zeros((horizon, actions, actions, states))
    mu = np.zeros((horizon, states, actions))
    nu = np.zeros((horizon, states, actions))
    for h in range(horizon - 1, -1, -1):
        for a1 in range(actions):
            for a2 in range(actions):
                expected = R[a1, a2] + P[a1, a2] @ V_max[h + 1]
                Q_max[a1, a2] = np.minimum(expected, horizon)

                expected = R[a1, a2] + P[a1, a2] @ V_min[h + 1]
                Q_min[a1, a2] = np.maximum(expected, 0)
        Q_opt[h] = Q_max

        for s in range(states):
            policy_s_max, policy_s_min = safe_equilibrium(Q_max[:, :, s], actions, seed=h * 7919 + s)
            V_max[h, s] = (policy_s_max[:, None] * Q_max[:, :, s] * policy_s_min[None, :]).sum()

            policy_i_max, policy_i_min = safe_equilibrium(Q_min[:, :, s], actions, seed=h * 7919 + s + 123)
            V_min[h, s] = (policy_i_max[:, None] * Q_min[:, :, s] * policy_i_min[None, :]).sum()

            mu[h, s] = policy_i_max
            nu[h, s] = policy_s_min

    return mu, nu, Q_opt


def generate_LCB(P_hat, R_hat, sigma, data_size):
    V_min = np.zeros((horizon + 1, states))
    V_max = np.zeros((horizon + 1, states))
    Q_min = np.zeros((actions, actions, states))
    Q_max = np.zeros((actions, actions, states))
    mu = np.zeros((horizon, states, actions))
    nu = np.zeros((horizon, states, actions))
    for h in range(horizon - 1, -1, -1):
        for a1 in range(actions):
            for a2 in range(actions):
                PV_min = alpha_max_backup(V_max[h + 1], P_hat[a1, a2], sigma)
                expected = R_hat[a1, a2] + PV_min
                Q_max[a1, a2] = np.minimum(expected, horizon)

                PV_max = alpha_max_backup(-V_min[h + 1], P_hat[a1, a2], sigma)
                expected = R_hat[a1, a2] - PV_max
                Q_min[a1, a2] = np.maximum(expected, 0)

        for s in range(states):
            policy_s_max, policy_s_min = safe_equilibrium(Q_max[:, :, s], actions, seed=h * 7919 + s)
            V_max[h, s] = (policy_s_max[:, None] * Q_max[:, :, s] * policy_s_min[None, :]).sum()

            policy_i_max, policy_i_min = safe_equilibrium(Q_min[:, :, s], actions, seed=h * 7919 + s + 123)
            V_min[h, s] = (policy_i_max[:, None] * Q_min[:, :, s] * policy_i_min[None, :]).sum()

            mu[h, s] = policy_i_max
            nu[h, s] = policy_s_min

    return mu, nu


def generate_LCB_Pes(P_hat, R_hat, R_count, sigma, cb, data_size):
    V_min = np.zeros((horizon + 1, states))
    V_max = np.zeros((horizon + 1, states))
    Q_min = np.zeros((actions, actions, states))
    Q_max = np.zeros((actions, actions, states))
    mu = np.zeros((horizon, states, actions))
    nu = np.zeros((horizon, states, actions))
    for h in range(horizon - 1, -1, -1):
        for a1 in range(actions):
            for a2 in range(actions):
                PV_min = alpha_max_backup(V_max[h + 1], P_hat[a1, a2], sigma)
                expected = R_hat[a1, a2] + PV_min
                b = bonus(P_hat[a1, a2], V_max[h + 1], R_count[a1, a2], cb, h)
                Q_max[a1, a2] = np.minimum(expected + b, horizon)

                PV_max = alpha_max_backup(-V_min[h + 1], P_hat[a1, a2], sigma)
                expected = R_hat[a1, a2] - PV_max
                b = bonus(P_hat[a1, a2], V_min[h + 1], R_count[a1, a2], cb, h)
                Q_min[a1, a2] = np.maximum(expected - b, 0)

        for s in range(states):
            policy_s_max, policy_s_min = safe_equilibrium(Q_max[:, :, s], actions, seed=h * 7919 + s)
            V_max[h, s] = (policy_s_max[:, None] * Q_max[:, :, s] * policy_s_min[None, :]).sum()

            policy_i_max, policy_i_min = safe_equilibrium(Q_min[:, :, s], actions, seed=h * 7919 + s + 123)
            V_min[h, s] = (policy_i_max[:, None] * Q_min[:, :, s] * policy_i_min[None, :]).sum()

            mu[h, s] = policy_i_max
            nu[h, s] = policy_s_min

    return mu, nu


def flat_to_pair_zerosum(P_flat, R_flat):
    S, A_joint, _ = P_flat.shape
    P_pair = np.zeros((actions, actions, S, S))
    R_pair = np.zeros((2, actions, actions, S))

    for j in range(A_joint):
        a1, a2 = divmod(j, actions)
        P_pair[a1, a2] = P_flat[:, j, :]
        R_pair[0, a1, a2] = R_flat[0, :, j]
        R_pair[1, a1, a2] = R_flat[1, :, j]
    return P_pair, R_pair


def figure_wrt_states(seed_num, sigma, cb, data_size):
    s_range = range(seed_num)
    diff_lcb_seeds_s = np.zeros((len(s_range), states))
    diff_pes_seeds_s = np.zeros((len(s_range), states))
    diff_pom_seeds_s = np.zeros((len(s_range), states))

    for s in s_range:
        print("seed - ", s)
        np.random.seed(s)
        rng = default_rng(seed=s)
        P, R = generate_transition()
        dataset = generate_dataset(P, R, dataset_size=data_size, sigma=sigma)

        P_hat, R_hat, R_count = estimate_dataset(dataset)

        xi = min(0.25, 4.0 / np.sqrt(data_size))

        mu_opt, nu_opt, Q_opt = generate_VI(P, R, sigma)
        mu_VI, nu_VI = generate_LCB(P_hat, R_hat, sigma, data_size)
        mu_LCB, nu_LCB = generate_LCB_Pes(P_hat, R_hat, R_count, sigma, cb, data_size)
        V_opt = evaluate_policies(mu_opt, nu_opt, P, R)
        V_VI = evaluate_policies(mu_VI, nu_VI, P, R)
        V_LCB = evaluate_policies(mu_LCB, nu_LCB, P, R)

        P_value = V_opt[0, :]
        P_hat_value = V_VI[0, :]
        diff = np.abs(P_value - P_hat_value)
        diff_lcb_seeds_s[s, :] = diff

        P_hat_value = V_LCB[0, :]
        diff = np.abs(P_value - P_hat_value)
        diff_pes_seeds_s[s, :] = diff

    colorbar = ["C2", "lightcoral", "C0", "mediumpurple"]
    fig = plt.figure()
    mean_pes_vals = np.mean(diff_pes_seeds_s, axis=0)
    std_pes_vals = np.std(diff_pes_seeds_s, axis=0)
    print(std_pes_vals)
    plt.plot(range(states), mean_pes_vals, color=colorbar[1], label="RTZ-VI-LCB (Ours)")
    plt.fill_between(
        range(states),
        mean_pes_vals - std_pes_vals,
        mean_pes_vals + std_pes_vals,
        facecolor=colorbar[1],
        alpha=0.25,
    )
    print(mean_pes_vals, std_pes_vals)

    mean_lcb_vals = np.mean(diff_lcb_seeds_s, axis=0)
    std_lcb_vals = np.std(diff_lcb_seeds_s, axis=0)
    print(std_lcb_vals)
    plt.plot(range(states), mean_lcb_vals, color=colorbar[2], label="RTZ-VI")
    plt.fill_between(
        range(states),
        mean_lcb_vals - std_lcb_vals,
        mean_lcb_vals + std_lcb_vals,
        facecolor=colorbar[2],
        alpha=0.25,
    )
    print(mean_lcb_vals, std_lcb_vals)

    plt.xlabel("Index of states", fontsize=20)
    plt.ylabel(r"$|V^{\star, \sigma^+}(s) - V^{\widehat{\mu}, \widehat{\nu}}(s)|$", fontsize=20)
    plt.legend(loc="upper left")
    plt.xlim([0, 50])
    plt.yscale("log")
    plt.grid()
    fig.savefig(f"./NIPS/offline_state_sigma{sigma}_cb{cb}_N{data_size}_xi{xi}.pdf", format="pdf", bbox_inches="tight")


def figure_wrt_samples(seed_num, sigma, cb):
    rho = 1 / (states) * np.ones(states)
    print(rho)

    n_range = [math.floor(pow(math.e, i)) for i in np.arange(5.0, 9.5, 1.0)]
    print("n_range", n_range)
    s_range = range(seed_num)
    diff_lcb_seeds = np.zeros((len(s_range), len(n_range)))
    diff_pes_seeds = np.zeros((len(s_range), len(n_range)))
    diff_pom_seeds = np.zeros((len(s_range), len(n_range)))
    for s in s_range:
        print("seed - ", s)
        diff_lcb_values = []
        diff_pes_values = []
        diff_pom_values = []
        P, R = generate_transition()
        for n in n_range:
            print("n - ", n)
            np.random.seed(s * 1000 + n)

            rng = default_rng(seed=s * 1000 + n)
            xi = min(0.25, 4.0 / np.sqrt(n))
            dataset = generate_dataset(P, R, dataset_size=n, sigma=sigma)
            P_hat, R_hat, R_count = estimate_dataset(dataset)

            mu_opt, nu_opt, Q_opt = generate_VI(P, R, sigma)
            mu_VI, nu_VI = generate_LCB(P_hat, R_hat, sigma, n)
            mu_LCB, nu_LCB = generate_LCB_Pes(P_hat, R_hat, R_count, sigma, cb, n)
            V_opt = evaluate_policies(mu_opt, nu_opt, P, R)
            V_VI = evaluate_policies(mu_VI, nu_VI, P, R)
            V_LCB = evaluate_policies(mu_LCB, nu_LCB, P, R)

            P_value = V_opt[0, :]
            P_hat_value = V_VI[0, :]
            diff = np.abs(P_value - P_hat_value)
            diff_lcb_values.append(np.dot(rho, diff))
            P_hat_value = V_LCB[0, :]
            diff = np.abs(P_value - P_hat_value)
            diff_pes_values.append(np.dot(rho, diff))

        diff_lcb_seeds[s, :] = diff_lcb_values
        diff_pes_seeds[s, :] = diff_pes_values

    plt.figure()
    plt.plot(n_range, diff_pes_values)
    plt.xlabel("K")
    plt.ylabel("value diff")

    fig = plt.figure()
    plt.plot(n_range, np.mean(diff_pes_seeds, axis=0))
    plt.xlabel("N")
    plt.ylabel(r"$|V^{\star, \sigma^+}(\rho) - V^{\widehat{\mu}, \widehat{\nu}}(\rho)|$")

    colorbar = ["C2", "lightcoral", "C0", "mediumpurple"]
    fig = plt.figure()

    start_show = 1
    end_show = len(n_range)

    mean_pes_vals = np.mean(diff_pes_seeds, axis=0)
    std_pes_vals = np.std(diff_pes_seeds, axis=0)
    print(std_pes_vals)
    plt.plot(
        n_range[start_show:end_show],
        mean_pes_vals[start_show:end_show],
        color=colorbar[2],
        label="RTZ-VI-LCB (Ours)",
    )
    plt.fill_between(
        n_range[start_show:end_show],
        mean_pes_vals[start_show:end_show] - std_pes_vals[start_show:end_show],
        mean_pes_vals[start_show:end_show] + std_pes_vals[start_show:end_show],
        facecolor=colorbar[2],
        alpha=0.25,
    )
    print(n_range[start_show:end_show], mean_pes_vals[start_show:end_show], std_pes_vals[start_show:end_show])

    mean_lcb_vals = np.mean(diff_lcb_seeds, axis=0)
    std_lcb_vals = np.std(diff_lcb_seeds, axis=0)
    print(std_lcb_vals)
    plt.plot(
        n_range[start_show:end_show],
        mean_lcb_vals[start_show:end_show],
        color=colorbar[1],
        label="RTZ-VI",
    )
    plt.fill_between(
        n_range[start_show:end_show],
        mean_lcb_vals[start_show:end_show] - std_lcb_vals[start_show:end_show],
        mean_lcb_vals[start_show:end_show] + std_lcb_vals[start_show:end_show],
        facecolor=colorbar[1],
        alpha=0.25,
    )
    print(n_range[start_show:end_show], mean_lcb_vals[start_show:end_show], std_lcb_vals[start_show:end_show])

    plt.xlabel("Sample size K", fontsize=20)
    plt.ylabel(r"$|V^{\star, \sigma^+}(\rho) - V^{\widehat{\mu}, \widehat{\nu}}(\rho)|$", fontsize=20)
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.grid()
    plt.xscale("log")
    fig.savefig(f"./NIPS/offline_N-compare_sigma{sigma}_cb{cb}_xi{xi}.pdf", format="pdf", bbox_inches="tight")

    colorbar = ["C2", "lightcoral", "C0", "mediumpurple"]
    fig = plt.figure()

    mean_pes_vals = np.mean(diff_pes_seeds, axis=0)
    std_pes_vals = np.std(diff_pes_seeds, axis=0)
    plt.plot(
        np.log(n_range[start_show:end_show]),
        np.log(mean_pes_vals[start_show:end_show]),
        color=colorbar[1],
        label="RTZ-VI-LCB (Ours)",
    )
    plt.fill_between(
        np.log(n_range[start_show:end_show]),
        np.log(mean_pes_vals[start_show:end_show] - std_pes_vals[start_show:end_show]),
        np.log(mean_pes_vals[start_show:end_show] + std_pes_vals[start_show:end_show]),
        facecolor=colorbar[1],
        alpha=0.25,
    )

    model = LinearRegression()
    model.fit(np.log(n_range[start_show:end_show]).reshape((-1, 1)), np.log(mean_pes_vals[start_show:end_show]))
    y_pred = model.intercept_ + model.coef_ * np.log(n_range[start_show:end_show]).reshape((-1, 1))
    print(model.coef_)
    plt.plot(np.log(n_range[start_show:end_show]), y_pred, color=colorbar[3], label=f"Linear: slope$={model.coef_}$")

    plt.xlabel("log(sample size K)", fontsize=20)
    plt.ylabel(r"$\log(|V^{\star, \sigma^+}(\rho) - V^{\widehat{\mu}, \widehat{\nu}}(\rho)|)$", fontsize=20)
    plt.legend(loc="upper right")
    plt.grid()

    fig.savefig(f"./NIPS/offline_N_sigma{sigma}_cb{cb}_xi{xi}.pdf", format="pdf", bbox_inches="tight")

    model = LinearRegression()
    model.fit(np.log(n_range[start_show:end_show]).reshape((-1, 1)), np.log(mean_pes_vals[start_show:end_show]))
    print(model.coef_)
    _ = model.intercept_ + model.coef_ * np.log(n_range[start_show:end_show]).reshape((-1, 1))


def evaluate_policies(mu, nu, P_true, R_true):
    V_eval = np.zeros((horizon + 1, states))
    for h in reversed(range(horizon)):
        for s in range(states):
            Q_val = np.zeros((actions, actions))
            for a1 in range(actions):
                for a2 in range(actions):
                    Q_val[a1, a2] = R_true[a1, a2, s] + P_true[a1, a2, s] @ V_eval[h + 1]
            V_eval[h, s] = mu[h, s].dot(Q_val).dot(nu[h, s])
    return V_eval


if __name__ == "__main__":
    P, R = generate_transition()
    V_norobust, policy_max_norobust, policy_min_norobust = generate_best(P, R)
    for sigma in sigma_all:
        for cb in cb_all:
            print("cb:", cb)
            figure_wrt_samples(seed_num, sigma, cb)
