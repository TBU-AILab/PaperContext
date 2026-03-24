#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    bounds = np.array(bounds)
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    # CMA-ES inspired algorithm with restarts
    def elapsed():
        return (datetime.now() - start).total_seconds()

    def clip(x):
        return np.clip(x, lower, upper)

    # Phase 1: Latin Hypercube Sampling for initial exploration
    n_init = min(100, max(20, dim * 10))
    best_x = None

    for i in range(n_init):
        if elapsed() >= max_time * 0.95:
            return best
        x = np.array([np.random.uniform(l, h) for l, h in zip(lower, upper)])
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()

    if best_x is None:
        best_x = (lower + upper) / 2.0

    # Phase 2: CMA-ES style optimization
    sigma = 0.3
    pop_size = max(10, 4 + int(3 * np.log(dim)))
    mu = pop_size // 2

    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights ** 2)

    c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
    cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

    mean = best_x.copy()
    ps = np.zeros(dim)
    pc = np.zeros(dim)

    if dim <= 100:
        C = np.eye(dim)
        use_full_cov = True
    else:
        diag_C = np.ones(dim)
        use_full_cov = False

    chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

    generation = 0
    stagnation_count = 0
    prev_best = best

    while elapsed() < max_time * 0.95:
        generation += 1

        if use_full_cov:
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            except:
                C = np.eye(dim)
                sqrt_C = np.eye(dim)

        solutions = []
        fitnesses = []

        for i in range(pop_size):
            if elapsed() >= max_time * 0.95:
                return best
            z = np.random.randn(dim)
            if use_full_cov:
                x = mean + sigma * (sqrt_C @ z)
            else:
                x = mean + sigma * (np.sqrt(diag_C) * z)
            x = clip(x)
            f = func(x)
            solutions.append(x)
            fitnesses.append(f)
            if f < best:
                best = f
                best_x = x.copy()

        idx = np.argsort(fitnesses)
        selected = [solutions[i] for i in idx[:mu]]

        old_mean = mean.copy()
        mean = np.sum([w * s for w, s in zip(weights, selected)], axis=0)

        diff = (mean - old_mean) / sigma

        if use_full_cov:
            inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ diff)
        else:
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (diff / np.sqrt(diag_C))

        h_sig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (generation + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0

        pc = (1 - cc) * pc + h_sig * np.sqrt(cc * (2 - cc) * mu_eff) * diff

        if use_full_cov:
            artmp = np.array([(s - old_mean) / sigma for s in selected])
            C = (1 - c1 - c_mu) * C + c1 * (np.outer(pc, pc) + (1 - h_sig) * cc * (2 - cc) * C)
            for i in range(mu):
                C += c_mu * weights[i] * np.outer(artmp[i], artmp[i])
        else:
            artmp = np.array([(s - old_mean) / sigma for s in selected])
            diag_C = (1 - c1 - c_mu) * diag_C + c1 * (pc ** 2 + (1 - h_sig) * cc * (2 - cc) * diag_C)
            for i in range(mu):
                diag_C += c_mu * weights[i] * artmp[i] ** 2
            diag_C = np.maximum(diag_C, 1e-20)

        sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
        sigma = min(sigma, np.max(upper - lower))

        if best >= prev_best:
            stagnation_count += 1
        else:
            stagnation_count = 0
            prev_best = best

        if stagnation_count > 20 + dim or sigma < 1e-15:
            sigma = 0.3
            mean = best_x.copy() + 0.1 * np.random.randn(dim) * (upper - lower)
            mean = clip(mean)
            if use_full_cov:
                C = np.eye(dim)
            else:
                diag_C = np.ones(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            stagnation_count = 0

    return best
