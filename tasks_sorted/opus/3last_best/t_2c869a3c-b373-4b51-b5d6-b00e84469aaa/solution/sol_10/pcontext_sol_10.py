#Looking at the results, the best algorithm (38.03) uses IPOP-CMA-ES with LHS initialization and Nelder-Mead refinement. The others that scored worse (54-215) had issues with time management, overly complex strategies, or insufficient exploration. Let me analyze and improve:
#
#Key improvements:
#1. **Active CMA-ES** with negative weights for worst solutions (faster convergence)
#2. **Better BIPOP budget allocation** between local and global restarts
#3. **Adaptive sigma initialization** based on restart context
#4. **Faster eigendecomposition scheduling**
#5. **Enhanced local search** with both Nelder-Mead and golden-section line search
#6. **Better stagnation handling** with TolX/TolUpSigma criteria
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    center = (lower + upper) / 2.0

    def elapsed():
        return (datetime.now() - start).total_seconds()
    def time_left():
        return max_time * 0.95 - elapsed()

    def eval_f(x):
        nonlocal best, best_x
        x_c = np.clip(x, lower, upper)
        f = func(x_c)
        if f < best:
            best = f
            best_x = x_c.copy()
        return f

    def mirror_bounds(x):
        xn = x.copy()
        for i in range(dim):
            r = ranges[i]
            if r <= 0:
                xn[i] = lower[i]
                continue
            xn[i] -= lower[i]
            p = 2 * r
            xn[i] = xn[i] % p
            if xn[i] > r:
                xn[i] = p - xn[i]
            xn[i] += lower[i]
        return np.clip(xn, lower, upper)

    # Phase 1: LHS initialization
    n_init = min(max(30 * dim, 200), 1000)
    pts = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        pts[:, i] = lower[i] + (perm + np.random.uniform(0, 1, n_init)) / n_init * ranges[i]

    init_fitnesses = []
    for i in range(n_init):
        if time_left() <= 0:
            return best
        f = eval_f(pts[i])
        init_fitnesses.append((f, i))

    eval_f(center)
    init_fitnesses.sort()
    top_k = min(10, len(init_fitnesses))
    starting_points = [pts[init_fitnesses[i][1]].copy() for i in range(top_k)]

    # Phase 2: IPOP-CMA-ES with restarts
    base_pop_size = max(4 + int(3 * np.log(dim)), 10)
    restart_count = 0
    pop_multiplier = 1
    small_budget = 0
    large_budget = 0

    while time_left() > 0.5:
        # BIPOP strategy
        if restart_count < len(starting_points):
            x0 = starting_points[restart_count].copy()
            pop_size = base_pop_size
            sigma = np.mean(ranges) / 4.0
            is_local = True
        elif small_budget <= large_budget and best_x is not None:
            # Small local restart
            scale = 0.005 + 0.002 * restart_count
            scale = min(scale, 0.05)
            x0 = best_x + np.random.randn(dim) * ranges * scale
            x0 = np.clip(x0, lower, upper)
            pop_size = base_pop_size
            sigma = np.mean(ranges) * scale * 2
            is_local = True
        else:
            # Large global restart with increasing population
            x0 = np.array([np.random.uniform(l, u) for l, u in bounds])
            pop_multiplier = min(pop_multiplier * 2, 16)
            pop_size = min(base_pop_size * pop_multiplier, 256)
            sigma = np.mean(ranges) / 3.0
            is_local = False

        mu = pop_size // 2
        weights_pos = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights_pos = weights_pos / np.sum(weights_pos)
        mu_eff = 1.0 / np.sum(weights_pos ** 2)

        # Active CMA: negative weights for worst solutions
        n_neg = pop_size - mu
        weights_neg = np.log(mu + 0.5) - np.log(np.arange(mu + 1, pop_size + 1))
        if n_neg > 0 and np.sum(np.abs(weights_neg)) > 0:
            weights_neg = weights_neg / np.sum(np.abs(weights_neg))
            mu_eff_neg = 1.0 / np.sum(weights_neg ** 2) if np.sum(weights_neg ** 2) > 0 else 1.0
        else:
            weights_neg = np.zeros(n_neg)
            mu_eff_neg = 1.0

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu_val = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Active CMA coefficient
        alpha_mu_neg = 1 + c1 / c_mu_val if c_mu_val > 0 else 1
        alpha_mu_eff_neg = 1 + 2 * mu_eff_neg / (mu_eff + 2)
        alpha_pos_def = (1 - c1 - c_mu_val) / (dim * c_mu_val) if c_mu_val > 0 else 0
        c_mu_neg = min(c_mu_val * 0.5, min(alpha_mu_neg, min(alpha_mu_eff_neg, alpha_pos_def)))
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        use_sep = dim > 80
        mean = x0.copy()
        sg = sigma

        if use_sep:
            diag_C = np.ones(dim)
        else:
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            invsqrtC = np.eye(dim)
            eigen_countdown = 0

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        gen = 0
        stag = 0
        prev_best_local = float('inf')
        best_local = float('inf')
        flat_count = 0
        evals_restart = 0
        max_gen = max(100, 200 + 50 * dim // pop_size)
        initial_sigma = sg

        while time_left() > 0.2 and gen < max_gen and stag < 30 + 15 * dim // pop_size and sg > 1e-15:
            # TolUpSigma check
            if sg > initial_sigma * 1e4:
                break

            if use_sep:
                sq = np.sqrt(np.maximum(diag_C, 1e-20))
                Z = np.random.randn(pop_size, dim)
                X = mean + sg * sq * Z
            else:
                if eigen_countdown <= 0:
                    try:
                        C = (C + C.T) / 2
                        evals_c, B = np.linalg.eigh(C)
                        evals_c = np.maximum(evals_c, 1e-20)
                        D = np.sqrt(evals_c)
                        invsqrtC = B @ np.diag(1.0 / D) @ B.T
                    except:
                        C = np.eye(dim)
                        B = np.eye(dim)
                        D = np.ones(dim)
                        invsqrtC = np.eye(dim)
                    eigen_countdown = max(1, int(1 / (c1 + c_mu_val) / dim / 10))
                eigen_countdown -= 1
                Z = np.random.randn(pop_size, dim)
                X = mean + sg * (Z @ (B * D).T)

            sols = []
            fits = []
            for k in range(pop_size):
                if time_left() <= 0.15:
                    return best
                xk = mirror_bounds(X[k])
                f = eval_f(xk)
                sols.append(xk)
                fits.append(f)
                if f < best_local:
                    best_local = f

            evals_restart += pop_size
            ix = np.argsort(fits)
            old_mean = mean.copy()
            sel = np.array([sols[ix[i]] for i in range(mu)])
            mean = weights_pos @ sel
            md = mean - old_mean

            if use_sep:
                sq_safe = np.maximum(sq, 1e-20)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * md / (sg * sq_safe)
            else:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * invsqrtC @ md / sg

            nps = np.linalg.norm(p_sigma)
            hs = 1 if nps / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            p_c = (1 - c_c) * p_c + hs * np.sqrt(c_c * (2 - c_c) * mu_eff) * md / sg

            if use_sep:
                diag_C = (1 - c1 - c_mu_val) * diag_C + c1 * (p_c ** 2 + (1 - hs) * c_c * (2 - c_c) * diag_C)
                for i in range(mu):
                    diag_C += c_mu_val * weights_pos[i] * ((sols[ix[i]] - old_mean) / sg) ** 2
                # Active update for separable
                if c_mu_neg > 0:
                    for i in range(n_neg):
                        j = ix[mu + i]
                        diag_C -= c_mu_neg * abs(weights_neg[i]) * ((sols[j] - old_mean) / sg) ** 2
                diag_C = np.maximum(diag_C, 1e-20)
            else:
                artmp_pos = np.array([(sols[ix[i]] - old_mean) / sg for i in range(mu)]).T
                C_new = (1 - c1 - c_mu_val) * C + c1 * (np.outer(p_c, p_c) + (1 - hs) * c_c * (2 - c_c) * C) + c_mu_val * (artmp_pos * weights_pos) @ artmp_pos.T
                # Active CMA negative update
                if c_mu_neg > 0 and n_neg > 0:
                    artmp_neg = np.array([(sols[ix[mu + i]] - old_mean) / sg for i in range(n_neg)]).T
                    neg_weights = np.abs(weights_neg)
                    C_new -= c_mu_neg * (artmp_neg * neg_weights) @ artmp_neg.T
                C = C_new
                # Ensure positive definiteness
                try:
                    min_eig = np.min(np.linalg.eigvalsh(C))
                    if min_eig < 1e-15:
                        C += (1e-15 - min_eig) * np.eye(dim)
                except:
                    C = np.eye(dim)

            sg *= np.exp((c_sigma / d_sigma) * (nps / chi_n - 1))
            sg = min(sg, np.mean(ranges) * 2)
            gen += 1

            if best_local < prev_best_local - 1e-10:
                stag = 0
                prev_best_local = best_local
            else:
                stag += 1
            if len(set(fits)) <= 1:
                flat_count += 1
                if flat_count > 5:
                    break
            else:
                flat_count = 0

        if is_local:
            small_budget += evals_restart
        else:
            large_budget += evals_restart
        restart_count += 1

    # Phase 3: Nelder-Mead refinement
    if best_x is not None and time_left() > 0.5:
        n = dim
        step = np.minimum(ranges * 0.01, np.abs(best_x) * 0.05 + 1e-4)
        simplex = [best_x.copy()]
        simplex_f = [best]
        for i in range(n):
            if time_left() <= 0.2:
                break
            xi = best_x.copy()
            xi[i] += step[i] if xi[i] + step[i] <= upper[i] else -step[i]
            simplex.append(xi)
            simplex_f.append(eval_f(xi))

        if len(simplex) == n + 1:
            nm_iter = 0
            while time_left() > 0.1 and nm_iter < 5000:
                nm_iter += 1
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                centroid = np.mean(simplex[:-1], axis=0)
                xr = mirror_bounds(centroid + 1.0 * (centroid - simplex[-1]))
                fr = eval_f(xr)
                if fr < simplex_f[0]:
                    xe = mirror_bounds(centroid + 2.0 * (xr - centroid))
                    fe = eval_f(xe)
                    simplex[-1], simplex_f[-1] = (xe, fe) if fe < fr else (xr, fr)
                elif fr < simplex_f[-2]:
                    simplex[-1], simplex_f[-1] = xr, fr
                else:
                    if fr < simplex_f[-1]:
                        xc = mirror
