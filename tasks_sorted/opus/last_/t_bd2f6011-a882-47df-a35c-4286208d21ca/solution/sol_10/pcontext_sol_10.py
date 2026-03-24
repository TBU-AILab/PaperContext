#
#
#Looking at the previous code, I see it was cut off and there are several areas for improvement. Let me create a more robust optimization algorithm with:
#
#1. Better Latin Hypercube sampling
#2. Improved L-SHADE with better parameter adaptation
#3. CMA-ES with proper restarts and sigma adaptation
#4. Nelder-Mead local refinement
#5. Better time management
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower

    def time_left():
        return max_time - (datetime.now() - start).total_seconds()

    def clip(x):
        return np.clip(x, lower, upper)

    evals = [0]
    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        evals[0] += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- Phase 1: Latin Hypercube Sampling ---
    n_init = min(1000, max(300, dim * 25))
    perms = np.zeros((n_init, dim))
    for d in range(dim):
        perms[:, d] = np.random.permutation(n_init)
    samples = (perms + np.random.uniform(0, 1, (n_init, dim))) / n_init
    init_pop = lower[None, :] + samples * ranges[None, :]
    init_fits = []
    for i in range(n_init):
        if time_left() < 1.0:
            return best
        init_fits.append(evaluate(init_pop[i]))
    init_fits = np.array(init_fits)

    # --- Phase 2: L-SHADE ---
    pop_size_init = min(150, max(50, dim * 8))
    pop_size_min = max(4, dim)

    sorted_init = np.argsort(init_fits)
    n_top = min(pop_size_init, n_init)
    pop = init_pop[sorted_init[:n_top]].copy()
    pop_f = init_fits[sorted_init[:n_top]].copy()
    if len(pop_f) < pop_size_init:
        extra = pop_size_init - len(pop_f)
        for _ in range(extra):
            x = lower + np.random.uniform(0, 1, dim) * ranges
            f = evaluate(x)
            pop = np.vstack([pop, x[None, :]])
            pop_f = np.append(pop_f, f)

    memory_size = 10
    MF = np.full(memory_size, 0.5)
    MCR = np.full(memory_size, 0.5)
    mem_idx = 0
    archive = []

    de_time_frac = 0.40
    de_deadline = time_left() * de_time_frac
    de_start = datetime.now()
    gen = 0

    while True:
        elapsed_de = (datetime.now() - de_start).total_seconds()
        if elapsed_de > de_deadline or time_left() < 2.0:
            break

        NP = len(pop_f)
        ratio = min(1.0, elapsed_de / max(de_deadline, 1e-9))
        target_NP = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * ratio)))
        if target_NP < NP:
            idx_keep = np.argsort(pop_f)[:target_NP]
            pop = pop[idx_keep]
            pop_f = pop_f[idx_keep]
            NP = target_NP

        S_F, S_CR, S_df = [], [], []
        trial_pop = np.empty_like(pop)
        trial_f = np.full(NP, np.inf)

        for i in range(NP):
            if time_left() < 1.5:
                break

            ri = np.random.randint(memory_size)
            Fi = -1
            for _ in range(10):
                Fi = np.random.standard_cauchy() * 0.1 + MF[ri]
                if Fi > 0:
                    break
            if Fi <= 0:
                Fi = 0.01
            Fi = min(Fi, 1.0)

            if MCR[ri] < 0:
                CRi = 0.0
            else:
                CRi = np.clip(np.random.randn() * 0.1 + MCR[ri], 0.0, 1.0)

            p_rate = max(2.0/NP, 0.05 + 0.15 * (1 - ratio))
            p = max(2, int(p_rate * NP))
            sorted_idx = np.argsort(pop_f)
            pbest = sorted_idx[np.random.randint(p)]

            r1 = i
            while r1 == i:
                r1 = np.random.randint(NP)

            pool_size = NP + len(archive)
            r2 = i
            att = 0
            while (r2 == i or r2 == r1) and att < 25:
                r2 = np.random.randint(pool_size)
                att += 1
            xr2 = pop[r2] if r2 < NP else archive[r2 - NP]

            mutant = pop[i] + Fi * (pop[pbest] - pop[i]) + Fi * (pop[r1] - xr2)

            jrand = np.random.randint(dim)
            mask = np.random.rand(dim) < CRi
            mask[jrand] = True
            trial = np.where(mask, mutant, pop[i])

            below = trial < lower
            above = trial > upper
            trial[below] = (lower[below] + pop[i][below]) / 2.0
            trial[above] = (upper[above] + pop[i][above]) / 2.0

            f_trial = evaluate(trial)
            trial_pop[i] = trial
            trial_f[i] = f_trial

            if f_trial < pop_f[i]:
                S_F.append(Fi)
                S_CR.append(CRi)
                S_df.append(pop_f[i] - f_trial)
                if len(archive) < NP:
                    archive.append(pop[i].copy())
                else:
                    archive[np.random.randint(len(archive))] = pop[i].copy()

        for i in range(NP):
            if trial_f[i] <= pop_f[i]:
                pop[i] = trial_pop[i]
                pop_f[i] = trial_f[i]

        if S_F:
            w = np.array(S_df)
            ws = w.sum()
            if ws > 0:
                w = w / ws
            else:
                w = np.ones(len(w)) / len(w)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            MF[mem_idx] = float(np.sum(w * sf**2) / max(np.sum(w * sf), 1e-30))
            MCR[mem_idx] = float(np.sum(w * scr))
            mem_idx = (mem_idx + 1) % memory_size

        gen += 1

    # --- Phase 3: CMA-ES with restarts ---
    def run_cmaes(x0, sigma0, time_budget):
        nonlocal best, best_params
        t0 = datetime.now()
        n = dim
        use_sep = (n > 60)

        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)

        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2.0 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))

        mean = x0.copy().astype(float)
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)

        if use_sep:
            C_diag = np.ones(n)
        else:
            C = np.eye(n)
            eigenvalues = np.ones(n)
            eigenvectors = np.eye(n)
            invsqrtC = np.eye(n)
            eigen_count = 0

        stag = 0
        best_gen = float('inf')
        generation = 0

        while True:
            if (datetime.now() - t0).total_seconds() > time_budget or time_left() < 0.3:
                break

            arz = np.random.randn(lam, n)
            if use_sep:
                sqrtC = np.sqrt(np.maximum(C_diag, 1e-20))
                arx = mean[None, :] + sigma * arz * sqrtC[None, :]
            else:
                arx = np.empty((lam, n))
                for k in range(lam):
                    arx[k] = mean + sigma * (eigenvectors @ (np.sqrt(np.maximum(eigenvalues, 1e-20)) * arz[k]))

            for k in range(lam):
                arx[k] = clip(arx[k])

            fits = np.array([evaluate(arx[k]) for k in range(lam)])
            if time_left() < 0.3:
                break

            order = np.argsort(fits)
            if fits[order[0]] < best_gen - 1e-12:
                best_gen = fits[order[0]]
                stag = 0
            else:
                stag += 1

            if stag > 15 + 10*n/lam:
                break

            old_mean = mean.copy()
            sel_x = arx[order[:mu]]
            mean = weights @ sel_x
            diff = (mean - old_mean) / sigma

            if use_sep:
                inv_sqrt = 1.0 / np.sqrt(np.maximum(C_diag, 1e-20))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt * diff)
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff)

            ps_norm = np.linalg.norm(ps)
            hsig = float(ps_norm**2 / (1 - (1-cs)**(2*(generation+1))) / n < 2 + 4.0/(n+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff

            artmp = (sel_x - old_mean[None, :]) / sigma

            if use_sep:
                C_diag = ((1 - c1 - cmu_val) * C_diag
                         + c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * C_diag)
                         + cmu_val * (weights[:, None] * artmp**2).sum(axis=0))
                C_diag = np.maximum(C_diag, 1e-20)
            else:
                C = ((1 - c1 - cmu_val) * C
                     + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                     + cmu_val * sum(weights[k] * np.outer(artmp[k], artmp[k]) for k in range(mu)))
                eigen_count += 1
                if eigen_count >= max(1, lam / (10 * n)):
                    C = np.triu(C) + np.triu(C, 1).T
                    try:
                        eigenvalues, eigenvectors = np.linalg.eigh(C)
                        eigenvalues = np.maximum(eigenvalues, 1e-20)
                        invsqrtC = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
                    except:
                        break
                    eigen_count = 0

            sigma *= np.exp((cs / damps) * (ps_norm / chiN - 1))
            sigma = np.clip(sigma, 1e-20, 1e3 * np.max(ranges))

            cond = sigma * np.sqrt(np.max(C_diag if use_sep else eigenvalues))
            if cond < 1e-14 * np.max(ranges):
                break
            generation += 1

    # Run CMA-ES restarts
    sorted_pop = np.argsort(pop_f)
    cma_starts = [pop[sorted_pop[i]].copy() for i in range(min(5, len(pop_f)))]

    restart = 0
    while time_left() > 1.5:
        budget = min(time_left() - 0.5, max(2.0, time_left() * 0.4))
        if restart < len(cma_starts):
            x0 = cma_starts[restart]
            sigma0 = 0.15 * np.max(ranges)
        elif restart < len(cma_starts) + 5:
            x0 = clip(best_params + np.random.randn(dim) * ranges * 0.02)
            sigma0 = 0.02 * np.max(ranges)
        else:
            x0 = lower + np.random.uniform(0, 1, dim) * ranges
            sigma0 = 0.3 * np.max(ranges)
        run_cmaes(x0, sigma0, budget)
        restart += 1

    # --- Phase 4: Nelder-Mead local search ---
    def nelder_mead(x0, step, budget):
        n = dim
        simplex = np.empty((n+1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i+1] = x0.copy()
            simplex[i+1][i] += step * ranges[i]
            simplex[i+1] = clip(simplex[i+1])
        fs = np.array([evaluate(simplex[i]) for i in range(n+1)])
        t0 = datetime.now()
        while (datetime.now()-t0).total_seconds() < budget and time_left() > 0.2:
            order = np.argsort(fs)
            simplex = simplex[order]
            fs = fs[order]
            centroid = simplex[:-1].mean(axis
