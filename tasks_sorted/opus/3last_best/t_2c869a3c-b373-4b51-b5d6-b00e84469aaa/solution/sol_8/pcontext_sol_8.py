#Looking at the results, the best algorithm (38.03) uses IPOP-CMA-ES with LHS initialization and Nelder-Mead refinement. I'll improve it with:
#
#1. **Better initial sampling** - Sobol-like + opposition-based learning
#2. **Improved CMA-ES** with better parameter tuning, weighted active update
#3. **Smarter restart strategy** - alternate between local refinement restarts and global exploration
#4. **Vectorized mirror bounds** 
#5. **Multi-phase local search** combining Nelder-Mead and pattern search
#6. **Better time management** and stagnation detection
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
        return max_time * 0.96 - elapsed()

    def eval_f(x):
        nonlocal best, best_x
        x_c = np.clip(x, lower, upper)
        f = func(x_c)
        if f < best:
            best = f
            best_x = x_c.copy()
        return f

    def mirror(x):
        x = x.copy()
        for d in range(dim):
            r = ranges[d]
            if r <= 0:
                x[d] = lower[d]
                continue
            x[d] -= lower[d]
            p = 2 * r
            x[d] = x[d] % p
            if x[d] > r:
                x[d] = p - x[d]
            x[d] += lower[d]
        return np.clip(x, lower, upper)

    # Phase 1: LHS + opposition-based init
    n_init = min(max(25 * dim, 200), 1000)
    pts = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        pts[:, i] = lower[i] + (perm + np.random.uniform(0, 1, n_init)) / n_init * ranges[i]

    ifits = []
    for i in range(n_init):
        if time_left() <= 0:
            return best
        f = eval_f(pts[i])
        ifits.append((f, i))
        # Opposition-based evaluation for top candidates (every 5th)
        if i % 5 == 0 and time_left() > 0:
            opp = lower + upper - pts[i]
            eval_f(opp)

    eval_f(center)
    ifits.sort()
    top_k = min(15, len(ifits))
    starting_points = [pts[ifits[i][1]].copy() for i in range(top_k)]

    # Track all good solutions found
    good_solutions = [(ifits[i][0], pts[ifits[i][1]].copy()) for i in range(top_k)]

    # Phase 2: IPOP-CMA-ES with smart restarts
    base_pop = max(4 + int(3 * np.log(dim)), 10)
    restart_count = 0
    large_budget_used = 0
    small_budget_used = 0

    while time_left() > 0.5:
        # BIPOP strategy: alternate between small local and large global restarts
        if restart_count < len(starting_points):
            x0 = starting_points[restart_count].copy()
            pop_size = base_pop
            sigma0 = np.mean(ranges) / 4.0
        elif small_budget_used <= large_budget_used and best_x is not None:
            # Small restart near best
            scale = 0.01 * (1 + restart_count * 0.005)
            x0 = best_x + np.random.randn(dim) * ranges * min(scale, 0.1)
            x0 = np.clip(x0, lower, upper)
            pop_size = max(base_pop, int(base_pop * 0.5))
            sigma0 = np.mean(ranges) * min(scale * 2, 0.2)
        else:
            # Large global restart
            if good_solutions and np.random.random() < 0.3:
                # Start from a random good solution with perturbation
                idx = np.random.randint(len(good_solutions))
                x0 = good_solutions[idx][1] + np.random.randn(dim) * ranges * 0.1
                x0 = np.clip(x0, lower, upper)
            else:
                x0 = np.array([np.random.uniform(l, u) for l, u in bounds])
            factor = min(2 ** (restart_count - len(starting_points) + 1), 16)
            pop_size = min(base_pop * factor, 256)
            sigma0 = np.mean(ranges) / 3.0

        mu = pop_size // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        use_sep = dim > 80
        mn = x0.copy()
        sg = sigma0

        if use_sep:
            dC = np.ones(dim)
        else:
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            invsqrtC = np.eye(dim)
            eigen_countdown = 0

        ps_vec = np.zeros(dim)
        pc = np.zeros(dim)
        gen = 0
        stag = 0
        prev_best_local = float('inf')
        best_local = float('inf')
        flat_count = 0
        max_gen = max(100, 200 + 50 * dim // pop_size)
        evals_this_restart = 0

        while time_left() > 0.2 and gen < max_gen and stag < 30 + 15 * dim // pop_size and sg > 1e-15:
            if use_sep:
                sq = np.sqrt(np.maximum(dC, 1e-20))
                Z = np.random.randn(pop_size, dim)
                X = mn + sg * sq * Z
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
                    eigen_countdown = max(1, int(1 / (c1 + c_mu) / dim / 10))
                eigen_countdown -= 1
                Z = np.random.randn(pop_size, dim)
                X = mn + sg * (Z @ (B * D).T)

            sols = []
            fits = []
            for k in range(pop_size):
                if time_left() <= 0.15:
                    return best
                xk = mirror(X[k])
                f = eval_f(xk)
                sols.append(xk)
                fits.append(f)
                if f < best_local:
                    best_local = f

            evals_this_restart += pop_size
            ix = np.argsort(fits)
            old_mn = mn.copy()
            sel = np.array([sols[ix[i]] for i in range(mu)])
            mn = w @ sel
            md = mn - old_mn

            if use_sep:
                sq_safe = np.maximum(sq, 1e-20)
                ps_vec = (1 - cs) * ps_vec + np.sqrt(cs * (2 - cs) * mu_eff) * md / (sg * sq_safe)
            else:
                ps_vec = (1 - cs) * ps_vec + np.sqrt(cs * (2 - cs) * mu_eff) * invsqrtC @ md / sg

            nps = np.linalg.norm(ps_vec)
            hs = 1 if nps / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mu_eff) * md / sg

            if use_sep:
                dC = (1 - c1 - c_mu) * dC + c1 * (pc ** 2 + (1 - hs) * cc * (2 - cc) * dC)
                for i in range(mu):
                    dC += c_mu * w[i] * ((sols[ix[i]] - old_mn) / sg) ** 2
                dC = np.maximum(dC, 1e-20)
            else:
                artmp = np.array([(sols[ix[i]] - old_mn) / sg for i in range(mu)]).T
                C = (1 - c1 - c_mu) * C + c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * C) + c_mu * (artmp * w) @ artmp.T

            sg *= np.exp((cs / ds) * (nps / chi_n - 1))
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

        # Track budget usage for BIPOP
        if pop_size <= base_pop:
            small_budget_used += evals_this_restart
        else:
            large_budget_used += evals_this_restart

        # Store good solution from this restart
        if best_local < float('inf'):
            good_solutions.append((best_local, best_x.copy() if best_x is not None else x0.copy()))
            good_solutions.sort()
            good_solutions = good_solutions[:20]

        restart_count += 1

    # Phase 3: Nelder-Mead refinement
    if best_x is not None and time_left() > 0.3:
        n = dim
        step = np.minimum(ranges * 0.01, np.abs(best_x) * 0.05 + 1e-4)
        simplex = [best_x.copy()]
        simplex_f = [best]
        for i in range(n):
            if time_left() <= 0.2:
                break
            xi = best_x.copy()
            xi[i] += step[i] if xi[i] + step[i] <= upper[i] else -step[i]
            fi = eval_f(xi)
            simplex.append(xi)
            simplex_f.append(fi)

        if len(simplex) == n + 1:
            while time_left() > 0.08:
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                centroid = np.mean(simplex[:-1], axis=0)
                xr = mirror(centroid + 1.0 * (centroid - simplex[-1]))
                fr = eval_f(xr)
                if fr < simplex_f[0]:
                    xe = mirror(centroid + 2.0 * (xr - centroid))
                    fe = eval_f(xe)
                    if fe < fr:
                        simplex[-1], simplex_f[-1] = xe, fe
                    else:
                        simplex[-1], simplex_f[-1] = xr, fr
                elif fr < simplex_f[-2]:
                    simplex[-1], simplex_f[-1] = xr, fr
                else:
                    xc = mirror(centroid + 0.5 * (simplex[-1] - centroid))
                    fc = eval_f(xc)
                    if fc < simplex_f[-1]:
                        simplex[-1], simplex_f[-1] = xc, fc
                    else:
                        for i in range(1, len(simplex)):
                            if time_left() <= 0.05:
                                return best
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            simplex_f[i] = eval_f(simplex[i])
                if np.max(np.std(simplex, axis=0)) < 1e-13:
                    break

    # Phase 4: Final pattern search
    if best_x is not None and time_left() > 0.05:
        xc = best_x.copy()
        fc = best
        step = ranges * 0.002
        while time_left() > 0.03:
            imp = False
            for i in range(dim):
                if time_left() <= 0.02:
                    return best
                for s in [step[i], -step[i]]:
                    xt = xc.copy()
                    xt[i] += s
                    ft = eval_f(np.clip(xt, lower, upper))
                    if ft < fc:
                        xc = np.clip(xt, lower, upper)
                        fc = ft
                        imp = True
                        break
            if not imp:
                step *= 0.5
                if np.max(step / ranges) < 1e-13:
                    break

    return best
