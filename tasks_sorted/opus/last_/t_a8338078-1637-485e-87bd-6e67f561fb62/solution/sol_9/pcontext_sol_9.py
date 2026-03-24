#Looking at the previous code, I notice several issues:
#1. There's a reference to `time_left()` which is undefined
#2. The elite CMA-ES phase has bugs
#3. Time allocation could be better
#4. The DE and CMA-ES phases could be more efficient
#5. Need better adaptive strategies
#
#Let me fix these and improve the overall algorithm:
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
    mid = (lower + upper) / 2.0

    def elapsed():
        return (datetime.now() - start).total_seconds()

    def time_ok(fraction=0.97):
        return elapsed() < max_time * fraction

    def remaining():
        return max(0, max_time - elapsed())

    eval_count = 0

    def eval_f(x):
        nonlocal best, best_x, eval_count
        x = np.clip(x, lower, upper)
        f = func(x)
        eval_count += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # LHS sampling
    def lhs_sample(n_samples):
        result = np.zeros((n_samples, dim))
        for d in range(dim):
            perm = np.random.permutation(n_samples)
            for i in range(n_samples):
                result[i, d] = lower[d] + ((perm[i] + np.random.random()) / n_samples) * ranges[d]
        return result

    # Phase 1: Initial sampling with LHS
    n_init = min(max(15 * dim, 80), 400)
    if dim > 50:
        n_init = min(n_init, 150)

    population = []
    samples = lhs_sample(n_init)
    for i in range(n_init):
        if not time_ok(0.04):
            break
        f = eval_f(samples[i])
        population.append((f, samples[i].copy()))

    eval_f(mid.copy())

    # Opposition-based learning
    population.sort(key=lambda t: t[0])
    n_opp = min(10, len(population))
    for i in range(n_opp):
        if not time_ok(0.06):
            break
        opp = lower + upper - population[i][1]
        f = eval_f(opp)
        population.append((f, opp.copy()))

    population.sort(key=lambda t: t[0])

    # Phase 2: Quick SHADE (Success-History based Adaptive DE)
    def shade_phase(pop, time_frac_end, max_gens=50):
        nonlocal best, best_x
        np_size = len(pop)
        pop = [list(p) for p in pop]
        
        H = 20  # memory size
        M_F = [0.5] * H
        M_CR = [0.8] * H
        k = 0
        
        archive = []
        max_archive = np_size
        
        for gen in range(max_gens):
            if not time_ok(time_frac_end):
                break
            
            S_F = []
            S_CR = []
            S_df = []
            
            trials = []
            for i in range(np_size):
                if not time_ok(time_frac_end):
                    break
                
                r = np.random.randint(H)
                Fi = np.clip(np.random.standard_cauchy() * 0.1 + M_F[r], 0, 1)
                CRi = np.clip(np.random.randn() * 0.1 + M_CR[r], 0, 1)
                
                # current-to-pbest/1
                p = max(2, int(0.1 * np_size))
                pbest_idx = np.random.randint(p)
                sorted_pop = sorted(range(np_size), key=lambda j: pop[j][0])
                x_pbest = pop[sorted_pop[pbest_idx]][1]
                
                idxs = list(range(np_size))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                
                # r2 from pop + archive
                combined = idxs + list(range(np_size, np_size + len(archive)))
                r2 = np.random.choice(combined)
                if r2 < np_size:
                    x_r2 = pop[r2][1]
                else:
                    x_r2 = archive[r2 - np_size]
                
                mutant = pop[i][1] + Fi * (x_pbest - pop[i][1]) + Fi * (pop[r1][1] - x_r2)
                
                trial = pop[i][1].copy()
                j_rand = np.random.randint(dim)
                mask = np.random.random(dim) < CRi
                mask[j_rand] = True
                trial[mask] = mutant[mask]
                trial = np.clip(trial, lower, upper)
                
                ft = eval_f(trial)
                trials.append((ft, trial, Fi, CRi, i))
            
            for ft, trial, Fi, CRi, i in trials:
                if ft < pop[i][0]:
                    df = pop[i][0] - ft
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    S_df.append(df)
                    if len(archive) < max_archive:
                        archive.append(pop[i][1].copy())
                    elif max_archive > 0:
                        archive[np.random.randint(max_archive)] = pop[i][1].copy()
                    pop[i] = [ft, trial]
            
            if S_F:
                w = np.array(S_df)
                w = w / (w.sum() + 1e-30)
                mean_F = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
                mean_CR = np.sum(w * np.array(S_CR))
                M_F[k] = mean_F
                M_CR[k] = mean_CR
                k = (k + 1) % H
        
        return [(p[0], p[1]) for p in pop]

    de_pop_size = min(max(20, 5 * dim), 100, len(population))
    de_pop = [(p[0], p[1].copy()) for p in population[:de_pop_size]]
    de_pop = shade_phase(de_pop, 0.18)
    de_pop.sort(key=lambda t: t[0])

    elite_pool = []
    for i in range(min(8, len(de_pop))):
        elite_pool.append((de_pop[i][0], de_pop[i][1].copy()))

    # CMA-ES implementation
    def cmaes_run(x0, sigma0, lam_override=None, max_time_abs=None):
        nonlocal best, best_x
        if max_time_abs is None:
            max_time_abs = elapsed() + 0.15 * max_time

        n = dim
        lam = lam_override if lam_override else max(6, 4 + int(3 * np.log(n)))
        mu = lam // 2

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)

        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

        xmean = np.clip(x0.copy(), lower, upper)
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)

        use_sep = (n > 40)

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
        best_local_x = xmean.copy()
        stag_count = 0
        f_history = []

        while elapsed() < max_time_abs and time_ok(0.96):
            if not use_sep and eigen_countdown <= 0:
                try:
                    C = np.triu(C) + np.triu(C, 1).T
                    eigvals, B = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    D = np.sqrt(eigvals)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
                    eigen_countdown = max(1, int(lam / (10 * n)))
                except:
                    return best_local_x
                if np.max(D) / (np.min(D) + 1e-30) > 1e14:
                    return best_local_x
            if not use_sep:
                eigen_countdown -= 1

            arxs = []
            arfitness = []

            for k in range(lam):
                if not time_ok(0.96) or elapsed() >= max_time_abs:
                    return best_local_x
                z = np.random.randn(n)
                if use_sep:
                    y = np.sqrt(diagC) * z
                else:
                    y = B @ (D * z)
                x = xmean + sigma * y
                x = np.clip(x, lower, upper)
                f = eval_f(x)
                arxs.append(x)
                arfitness.append(f)

            idx = np.argsort(arfitness)

            if arfitness[idx[0]] < best_local:
                best_local = arfitness[idx[0]]
                best_local_x = arxs[idx[0]].copy()
                stag_count = 0
            else:
                stag_count += 1

            f_history.append(arfitness[idx[0]])

            xold = xmean.copy()
            xmean = np.zeros(n)
            for i in range(mu):
                xmean += weights[i] * arxs[idx[i]]

            diff = xmean - xold

            if use_sep:
                inv_sqrt_diag = 1.0 / (np.sqrt(diagC) + 1e-30)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_diag * diff) / max(sigma, 1e-30)
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
                artmp_pos = np.zeros((mu, n))
                for i in range(mu):
                    artmp_pos[i] = (arxs[idx[i]] - xold) / max(sigma, 1e-30)
                rank_mu_pos = np.zeros((n, n))
                for i in range(mu):
                    rank_mu_pos += weights[i] * np.outer(artmp_pos[i], artmp_pos[i])
                C = (1 - c1 - cmu_val) * C + \
                    c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
                    cmu_val * rank_mu_pos

            sigma *= np.exp((cs / damps) * (ps_norm / chiN - 1))
            sigma = np.clip(sigma, 1e-20, 5 * np.max(ranges))

            gen += 1

            if sigma < 1e-16 * np.mean(ranges):
                return best_local_x
            if stag_count > 15 + 10 * n / lam:
                return best_local_x
            if gen > 300 + 200 * n / lam:
                return best_local_x

            if len(f_history) > 30:
                recent = f_history[-30:]
                if abs(max(recent) - min(recent)) < 1e-14 * (abs(recent[0]) + 1e-30):
                    return best_local_x

        return best_local_x

    default_lam = max(6, 4 + int(3 * np.log(dim)))

    # CMA-ES from top elite points
    n_elite_starts = min(3, len(elite_pool))
    for i in range(n_elite_starts):
        if not time_ok(0.25):
            break
        rem = remaining()
        x0 = elite_pool[i][1].copy()
        if i == 0 and best_x is not None:
            x0 = best_x.copy()
        sig0 = 0.15 * np.mean(ranges) * (0.3 + 0.7 * np.random.random())
        t_budget = elapsed() + min(rem * 0.15, rem / max(n_elite_starts - i, 1))
        cmaes_run(x0, sig0, max_time_abs=t_budget)

    # BIPOP restarts
    large_budget = 0.0
    small_budget = 0.0
    restart = 0

    while time_ok(0.80):
        rem = remaining()
        if rem < 0.3:
            break

        use_large = (large_budget <= small_budget) or np.random.random() < 0.3

        if use_large:
            power = min(restart + 1, 8)
            large_lam = int(default_lam * (2 ** np.random.uniform(0, power)))
            large_lam = min(large_lam, max(512, 8 * dim))
            large_lam = max(large_lam, default_lam)

            if np.random.random() < 0.35 and best_x is not None:
                x0 = best_x + np.random.randn(dim) * 0.3 * ranges
                x0 = np.clip(x0, lower, upper)
            else:
                x0 = lower + np.random.uniform(0, 1, dim) * ranges
            sig0 = (0.2 + 0.3 * np.random.random()) * np.mean(ranges)

            t_start = elapsed()
            t_budget = elapsed() + min(rem * 0.25, rem)
            cmaes_run(x0, sig0, lam_override=large_lam, max_time_abs=t_budget)
            large_budget += elapsed() - t_start
        else:
            small_lam = max(6, int(default_lam * (0.5 + 0.5 * np.random.random())))

            if best_x is not None and np.random.random() < 0.85:
                spread = 0.002 + 0.08 * np.random.random()
                x0 = best_x + np.random.randn(dim) * spread * ranges
                x0 = np.clip(x0, lower, upper)
                sig0 = (0.003 + 0.05 * np.random.random()) * np.mean(ranges)
            else:
                x0 = lower + np.random.uniform(0, 1, dim) * ranges
                sig0 = 0.15 * np.mean(ranges)

            t_start = elapsed()
            t_budget = elapsed() + min(rem * 0.08, rem)
            cmaes_run(x0, sig0, lam_override=small_lam, max_time_abs=t_budget)
            small_budget += elapsed() - t_start

        restart += 1

    # Nelder-Mead simplex for local refinement
    if best_x is not None and time_ok(0.88):
        n = dim
        alpha_nm, gamma_nm, rho_nm, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        # Initialize simplex around best
        scale = 0.01 * ranges
        simplex = [best_x.copy()]
        simplex_f = [best]
        for i in range(n):
            if not time_ok(0.89):
                break
            v = best_x.copy()
            v[i] += scale[i]
            v = np.clip(v, lower, upper)
            fv = eval_f(v)
            simplex.append(v)
            simplex_f.append(fv)
        
        if len(simplex) == n + 1:
            for iteration in range(200 * n):
                if not time_ok(0.94):
                    break
                
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflect
                xr = centroid + alpha_nm * (centroid - simplex[-1])
                xr = np.clip(xr, lower, upper)
                fr = eval_f(xr)
                
                if simplex_f[0] <= fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                elif fr < simplex_f[0]:
                    # Expand
                    xe = centroid + gamma_nm * (xr - centroid)
                    xe = np.clip(xe, lower, upper)
                    fe = eval_f(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                else:
                    # Contract
                    if fr < simplex_f[-1]:
                        xc = centroid + rho_nm * (xr - centroid)
                    else:
                        xc = centroid + rho_nm * (simplex[-1] - centroid)
                    xc = np.clip(xc, lower, upper)
                    fc = eval_f(xc)
                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            if not time_ok(0.94):
                                break
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            simplex_f[i] = eval_f(simplex[i])
                
                # Check convergence
                if max(simplex_f) - min(simplex_f) < 1e-15 * (abs(min(simplex_f)) + 1e-30):
                    break

    # Fine coordinate descent with adaptive step
    if best_x is not None and time_ok(0.95):
        x_curr = best_x.copy()
        f_curr = best

        for scale_exp in range(-3, -12, -1):
            if not time_ok(0.995):
                break
            step_size = 10**scale_exp * np.mean(ranges)
            improved = True
            passes = 0
            while improved and time_ok(0.995) and passes < 3:
                improved = False
                passes += 1
                for i in np.random.permutation(dim):
                    if not time_ok(0.998):
                        break
                    for direction in [1, -1]:
                        x_trial = x_curr.copy()
                        x_trial[i] = np.clip(x_curr[i] + direction * step_size, lower[i], upper[i])
                        ft = eval_f(x_trial)
                        if ft < f_curr:
                            f_curr = ft
                            x_curr = x_trial
                            improved = True
                            break

    return best
