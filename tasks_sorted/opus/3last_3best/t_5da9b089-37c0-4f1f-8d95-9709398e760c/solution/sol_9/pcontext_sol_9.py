#Looking at the results, algorithm 3 (simple CMA-ES with restarts, score 1.098) was the best. The key issues with the other algorithms were: over-complicated restart strategies, poor time allocation, and bugs in CMA-ES implementation. I'll create a cleaner, more robust CMA-ES with:
#
#1. **Proper BIPOP-CMA-ES** (alternating small/large population restarts)
#2. **Efficient eigendecomposition** (only when needed)
#3. **Better stagnation detection**
#4. **Smarter time allocation**
#5. **Final Nelder-Mead + coordinate descent polish**
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None

    bounds_array = np.array(bounds)
    lower = bounds_array[:, 0]
    upper = bounds_array[:, 1]
    ranges = upper - lower

    def elapsed():
        return (datetime.now() - start).total_seconds()

    def clamp(x):
        return np.clip(x, lower, upper)

    def evaluate(x):
        nonlocal best, best_params
        x = clamp(x)
        val = func(x)
        if val < best:
            best = val
            best_params = x.copy()
        return val

    # Initial sampling
    n_init = min(40 * dim, max(100, 15 * dim))
    for i in range(n_init):
        if elapsed() >= max_time * 0.10:
            break
        x = lower + np.random.rand(dim) * ranges
        evaluate(x)

    def run_cmaes(mean_init, sigma_init, lam_mult, time_limit):
        nonlocal best, best_params
        n = dim
        lam = max(6, int((4 + int(3 * np.log(n))) * lam_mult))
        # Make lambda even
        lam = lam + (lam % 2)
        mu = lam // 2

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        c_mu_val = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

        mean = mean_init.copy()
        sigma = sigma_init
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        use_full = n <= 200

        if use_full:
            C = np.eye(n)
            eigvals = np.ones(n)
            eigvecs = np.eye(n)
            D = np.ones(n)
            inv_D = np.ones(n)
            eigen_counter = 0
            eigen_interval = max(1, int(lam / (c1 + c_mu_val) / n / 10))
        else:
            diag_C = np.ones(n)

        gen = 0
        stag = 0
        local_best = float('inf')
        best_f_hist = []
        median_f_hist = []

        while elapsed() < time_limit:
            gen += 1

            if use_full:
                eigen_counter += 1
                if eigen_counter >= eigen_interval or gen == 1:
                    try:
                        C = (C + C.T) / 2.0
                        raw_eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(raw_eigvals, 1e-20)
                        D = np.sqrt(eigvals)
                        inv_D = 1.0 / D
                        eigen_counter = 0
                    except:
                        C = np.eye(n)
                        eigvals = np.ones(n)
                        eigvecs = np.eye(n)
                        D = np.ones(n)
                        inv_D = np.ones(n)

            offspring = np.zeros((lam, n))
            offspring_f = np.full(lam, float('inf'))

            for i in range(lam):
                if elapsed() >= time_limit:
                    return
                z = np.random.randn(n)
                if use_full:
                    y = eigvecs @ (D * z)
                else:
                    y = np.sqrt(diag_C) * z
                offspring[i] = clamp(mean + sigma * y)
                offspring_f[i] = evaluate(offspring[i])

            idx = np.argsort(offspring_f)
            old_mean = mean.copy()
            selected = offspring[idx[:mu]]
            mean = np.sum(weights[:, None] * selected, axis=0)

            md = (mean - old_mean) / sigma

            if use_full:
                inv_sqrt_md = eigvecs @ (inv_D * (eigvecs.T @ md))
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * inv_sqrt_md
            else:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (md / np.sqrt(diag_C))

            ps_norm = np.linalg.norm(p_sigma)
            h_sig = 1.0 if ps_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * gen)) < (1.4 + 2.0 / (n + 1)) * chi_n else 0.0

            p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * md

            if use_full:
                artmp = (selected - old_mean) / sigma
                rank_mu = sum(weights[i] * np.outer(artmp[i], artmp[i]) for i in range(mu))
                C = (1 - c1 - c_mu_val) * C + c1 * (np.outer(p_c, p_c) + (1 - h_sig) * c_c * (2 - c_c) * C) + c_mu_val * rank_mu
            else:
                artmp = (selected - old_mean) / sigma
                diag_C = (1 - c1 - c_mu_val) * diag_C + c1 * (p_c**2 + (1 - h_sig) * c_c * (2 - c_c) * diag_C)
                for i in range(mu):
                    diag_C += c_mu_val * weights[i] * artmp[i]**2
                diag_C = np.maximum(diag_C, 1e-20)

            sigma *= np.exp((c_sigma / d_sigma) * (ps_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-20, np.max(ranges) * 2)

            gb = offspring_f[idx[0]]
            if gb < local_best - 1e-12 * max(1, abs(local_best)):
                local_best = gb
                stag = 0
            else:
                stag += 1

            if stag > 10 + 30 * n // lam:
                return
            if sigma < 1e-16 * np.mean(ranges):
                return
            if use_full and np.max(D) / (np.min(D) + 1e-30) > 1e14:
                return

    # BIPOP restarts
    pop_mult = 1.0
    small_budget_used = 0
    large_budget_used = 0
    restart = 0

    while elapsed() < max_time * 0.95:
        restart += 1
        remaining = max_time * 0.95 - elapsed()
        t_alloc = max(remaining * 0.08, min(remaining, max_time * 0.06))
        tl = elapsed() + t_alloc

        if restart <= 2 and best_params is not None:
            run_cmaes(best_params.copy(), np.mean(ranges) * 0.2, 1.0, tl)
        elif small_budget_used <= large_budget_used:
            if best_params is not None:
                perturb = np.random.randn(dim) * ranges * 0.01 * (1 + restart * 0.5)
                m0 = clamp(best_params + perturb)
            else:
                m0 = lower + np.random.rand(dim) * ranges
            run_cmaes(m0, np.mean(ranges) * 0.05, 0.5, tl)
            small_budget_used += t_alloc
        else:
            m0 = lower + np.random.rand(dim) * ranges
            pop_mult = min(pop_mult * 2, 10)
            run_cmaes(m0, np.mean(ranges) * 0.3, pop_mult, tl)
            large_budget_used += t_alloc

    # Coordinate descent polish
    if best_params is not None and elapsed() < max_time * 0.995:
        x_c = best_params.copy()
        f_c = best
        step = 0.001 * ranges
        for _ in range(500):
            if elapsed() >= max_time * 0.995:
                break
            imp = False
            for d in range(dim):
                if elapsed() >= max_time * 0.995:
                    break
                for s in [1, -1]:
                    xt = x_c.copy()
                    xt[d] += s * step[d]
                    ft = evaluate(clamp(xt))
                    if ft < f_c:
                        x_c, f_c = xt, ft
                        step[d] *= 1.5
                        imp = True
                        break
            if not imp:
                step *= 0.5
                if np.max(step / (ranges + 1e-30)) < 1e-15:
                    break

    return best
