#Looking at the previous algorithm, I can identify several areas for improvement:
#
#1. The CMA-ES budget allocation is suboptimal - too much time on initial samples, poor restart scheduling
#2. Need better initial sampling with opposition-based learning
#3. The local search phase needs improvement - add a proper L-BFGS-like gradient estimation
#4. Better adaptive restart strategy (BIPOP instead of just IPOP)
#
#Let me implement a more refined version:
#
import numpy as np
from datetime import datetime, timedelta

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

    def time_ok(fraction=0.97):
        return elapsed() < max_time * fraction

    def eval_f(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # Phase 1: Latin Hypercube + opposition-based sampling
    n_init = min(max(15 * dim, 80), 500)
    if dim > 50:
        n_init = min(n_init, 150)

    init_points = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        init_points[:, i] = (perm + np.random.uniform(0, 1, n_init)) / n_init * ranges[i] + lower[i]

    population = []
    for i in range(n_init):
        if not time_ok(0.08):
            break
        f = eval_f(init_points[i])
        population.append((f, init_points[i].copy()))
        # Opposition-based point
        if time_ok(0.08):
            opp = lower + upper - init_points[i]
            fo = eval_f(opp)
            population.append((fo, opp.copy()))

    # Evaluate center
    if time_ok(0.09):
        eval_f(center.copy())

    population.sort(key=lambda t: t[0])

    # Phase 2: CMA-ES with BIPOP restarts
    def cmaes_run(x0, sigma0, lam_override=None, max_time_abs=None):
        nonlocal best, best_x

        if max_time_abs is None:
            max_time_abs = elapsed() + 0.2 * max_time

        n = dim
        if lam_override:
            lam = lam_override
        else:
            lam = 4 + int(3 * np.log(n))
        lam = max(lam, 6)
        if lam % 2 == 1:
            lam += 1
        mu = lam // 2

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)

        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

        xmean = np.clip(x0.copy(), lower, upper)
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)

        use_sep = (n > 60)

        if use_sep:
            diagC = np.ones(n)
        else:
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            invsqrtC = np.eye(n)
            eigen_countdown = 0

        gen = 0
        best_local = float('inf')
        stag_count = 0
        flat_count = 0
        f_history = []

        while time_ok(0.93) and elapsed() < max_time_abs:
            # Eigendecomposition
            if not use_sep and eigen_countdown <= 0:
                try:
                    C = np.triu(C) + np.triu(C, 1).T
                    eigvals, B = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    D = np.sqrt(eigvals)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                    eigen_countdown = max(1, lam // (10 * n))
                except:
                    return
            if not use_sep:
                eigen_countdown -= 1

            arxs = []
            arfitness = []
            arz = []

            for k in range(lam):
                if not time_ok(0.93) or elapsed() >= max_time_abs:
                    return

                z = np.random.randn(n)
                if use_sep:
                    y = np.sqrt(diagC) * z
                else:
                    y = B @ (D * z)

                x = xmean + sigma * y
                # Bounce boundary handling
                for d_i in range(n):
                    lo, hi = lower[d_i], upper[d_i]
                    while x[d_i] < lo or x[d_i] > hi:
                        if x[d_i] < lo:
                            x[d_i] = 2 * lo - x[d_i]
                        if x[d_i] > hi:
                            x[d_i] = 2 * hi - x[d_i]
                x = np.clip(x, lower, upper)

                f = eval_f(x)
                arxs.append(x)
                arfitness.append(f)
                arz.append(z)

            idx = np.argsort(arfitness)
            local_best = arfitness[idx[0]]

            f_history.append(local_best)
            if local_best < best_local - 1e-12 * (abs(best_local) + 1):
                best_local = local_best
                stag_count = 0
                flat_count = 0
            else:
                stag_count += 1

            frange = abs(arfitness[idx[0]] - arfitness[idx[-1]])
            if frange < 1e-15 * (abs(arfitness[idx[0]]) + 1e-30):
                flat_count += 1
                if flat_count > 5:
                    return

            xold = xmean.copy()
            xmean = np.zeros(n)
            for i in range(mu):
                xmean += weights[i] * arxs[idx[i]]

            diff = xmean - xold

            if use_sep:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * diff / (np.sqrt(diagC) * max(sigma, 1e-30) + 1e-30)
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff) / max(sigma, 1e-30)

            ps_norm = np.linalg.norm(ps)
            hsig = int(ps_norm / np.sqrt(1 - (1-cs)**(2*(gen+1))) / chiN < 1.4 + 2/(n+1))

            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff / max(sigma, 1e-30)

            if use_sep:
                artmp_sq = np.zeros(n)
                for i in range(mu):
                    d_vec = (arxs[idx[i]] - xold) / max(sigma, 1e-30)
                    artmp_sq += weights[i] * d_vec**2
                diagC = (1 - c1 - cmu_val) * diagC + \
                        c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * diagC) + \
                        cmu_val * artmp_sq
                diagC = np.maximum(diagC, 1e-20)
            else:
                artmp = np.zeros((mu, n))
                for i in range(mu):
                    artmp[i] = (arxs[idx[i]] - xold) / max(sigma, 1e-30)
                rank_mu_update = np.zeros((n, n))
                for i in range(mu):
                    rank_mu_update += weights[i] * np.outer(artmp[i], artmp[i])
                C = (1 - c1 - cmu_val) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu_val * rank_mu_update

            sigma *= np.exp((cs / damps) * (ps_norm / chiN - 1))
            sigma = np.clip(sigma, 1e-20, 5 * np.max(ranges))

            gen += 1

            if sigma < 1e-16:
                return
            if stag_count > 20 + 10 * n / lam:
                return
            if gen > 300 + 150 * n / lam:
                return

            if not use_sep and gen % 10 == 0:
                try:
                    cond = np.max(D) / (np.min(D) + 1e-30)
                    if cond > 1e14:
                        return
                except:
                    pass

            # Adaptive TPA-like sigma adaptation check
            if len(f_history) > 20:
                recent = f_history[-20:]
                if abs(recent[-1] - recent[0]) < 1e-15 * (abs(recent[0]) + 1e-30):
                    return

    # BIPOP restart strategy
    default_lam = 4 + int(3 * np.log(dim))
    large_lam = default_lam
    small_budget_used = 0
    large_budget_used = 0
    restart = 0
    num_good = min(5, len(population))

    while time_ok(0.90):
        rem_time = max_time * 0.93 - elapsed()
        if rem_time < 0.5:
            break

        if restart < num_good and restart < len(population):
            x0 = population[restart][1].copy()
            sig0 = 0.2 * np.mean(ranges)
            if restart == 0 and best_x is not None:
                x0 = best_x.copy()
                sig0 = 0.15 * np.mean(ranges)
            t_budget = elapsed() + min(rem_time * 0.3, rem_time / max(num_good - restart, 1))
            cmaes_run(x0, sig0, max_time_abs=t_budget)
        else:
            # BIPOP: alternate large and small population restarts
            use_large = (large_budget_used <= small_budget_used) or np.random.random() < 0.3
            
            if use_large:
                large_lam = int(default_lam * (2 ** min(restart - num_good + 1, 5)))
                large_lam = min(large_lam, max(512, 10 * dim))
                
                if np.random.random() < 0.3 and best_x is not None:
                    x0 = best_x + np.random.randn(dim) * 0.3 * ranges
                    x0 = np.clip(x0, lower, upper)
                else:
                    x0 = lower + np.random.uniform(0, 1, dim) * ranges
                sig0 = 0.3 * np.mean(ranges)
                
                t_start = elapsed()
                t_budget = elapsed() + min(rem_time * 0.35, rem_time)
                cmaes_run(x0, sig0, lam_override=large_lam, max_time_abs=t_budget)
                large_budget_used += elapsed() - t_start
            else:
                small_lam = max(6, int(default_lam * (0.5 + np.random.random())))
                
                if best_x is not None and np.random.random() < 0.5:
                    x0 = best_x + np.random.randn(dim) * (0.01 + 0.2 * np.random.random()) * ranges
                    x0 = np.clip(x0, lower, upper)
                    sig0 = (0.05 + 0.15 * np.random.random()) * np.mean(ranges)
                else:
                    x0 = lower + np.random.uniform(0, 1, dim) * ranges
                    sig0 = 0.2 * np.mean(ranges)
                
                t_start = elapsed()
                t_budget = elapsed() + min(rem_time * 0.2, rem_time)
                cmaes_run(x0, sig0, lam_override=small_lam, max_time_abs=t_budget)
                small_budget_used += elapsed() - t_start

        restart += 1

    # Phase 3: Gradient-estimated local search
    if best_x is not None and time_ok(0.95):
        x_curr = best_x.copy()
        f_curr = best
        
        for scale in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
            if not time_ok(0.97):
                break
            step = scale * ranges
            improved = True
            while improved and time_ok(0.97):
                improved = False
                # Estimate gradient
                grad = np.zeros(dim)
                skip_grad = False
                for i in range(dim):
                    if not time_ok(0.97):
                        skip_grad = True
                        break
                    xp = x_curr.copy()
                    xp[i] = min(x_curr[i] + step[i], upper[i])
                    fp = eval_f(xp)
                    xm = x_curr.copy()
                    xm[i] = max(x_curr[i] - step[i], lower[i])
                    fm = eval_f(xm)
                    grad[i] = (fp - fm) / (xp[i] - xm[i] + 1e-30)
                    # Opportunistic coordinate descent
                    bf = min(fp, fm)
                    if bf < f_curr:
                        f_curr = bf
                        x_curr = xp if fp < fm else xm
                        improved = True
                
                if skip_grad:
                    break
                
                # Line search along negative gradient
                gnorm = np.linalg.norm(grad)
                if gnorm > 1e-30 and time_ok(0.97):
                    direction = -grad / gnorm
                    for alpha in [0.1 * scale, scale, 5 * scale]:
                        if not time_ok(0.97):
                            break
                        x_trial = np.clip(x_curr + alpha * np.mean(ranges) * direction, lower, upper)
                        ft = eval_f(x_trial)
                        if ft < f_curr:
                            f_curr = ft
                            x_curr = x_trial
                            improved = True

    # Phase 4: Final coordinate-wise golden section refinement
    if best_x is not None and time_ok(0.98):
        x_curr = best_x.copy()
        f_curr = best
        gr = (np.sqrt(5) + 1) / 2
        
        for i in range(dim):
            if not time_ok(0.995):
                break
            a = max(lower[i], x_curr[i] - 0.001 * ranges[i])
            b = min(upper[i], x_curr[i] + 0.001 * ranges[i])
            
            for _ in range(15):
                if not time_ok(0.995) or (b - a) < 1e-15:
                    break
                c = b - (b - a) / gr
                d = a + (b - a) / gr
                xc = x_curr.copy(); xc[i] = c
                xd = x_curr.copy(); xd[i] = d
                fc = eval_f(xc)
                fd = eval_f(xd)
                if fc < fd:
                    b = d
                else:
                    a = c
            
            mid = (a + b) / 2
            xm = x_curr.copy(); xm[i] = mid
            fm = eval_f(xm)
            if fm < f_curr:
                f_curr = fm
                x_curr = xm

    return best
