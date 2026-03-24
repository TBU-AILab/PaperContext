#Looking at the two algorithms, the best one (score 3.19) uses L-SHADE + coordinate descent + Nelder-Mead, while the CMA-ES version scored 156.8. I'll create an improved hybrid that combines the strengths of both approaches with key improvements:
#
#1. **Better L-SHADE** with improved mutation strategies (current-to-pbest/1 with weighted recombination)
#2. **CMA-ES as a secondary optimizer** on the best region found
#3. **More aggressive restarts** with diverse starting points
#4. **Golden section search** for local refinement along each coordinate
#5. **Better time allocation**
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower

    def elapsed():
        return (datetime.now() - start).total_seconds()

    def clip(x):
        return np.clip(x, lower, upper)

    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    def lshade_run(time_limit, pop_init=None):
        nonlocal best, best_params

        N_init = min(max(30, 8 * dim), 250)
        N_min = 4

        if pop_init is not None:
            population = pop_init.copy()
            N_init = len(population)
        else:
            population = np.zeros((N_init, dim))
            for d in range(dim):
                perm = np.random.permutation(N_init)
                for i in range(N_init):
                    population[i, d] = lower[d] + (perm[i] + np.random.random()) / N_init * ranges[d]

        pop_size = len(population)
        fitness = np.array([evaluate(population[i]) for i in range(pop_size) if elapsed() < time_limit])
        if len(fitness) < pop_size:
            fitness = np.append(fitness, [float('inf')] * (pop_size - len(fitness)))

        if pop_init is None and elapsed() < time_limit:
            opp = lower + upper - population
            opp_fitness = np.array([evaluate(opp[i]) for i in range(pop_size) if elapsed() < time_limit])
            if len(opp_fitness) == pop_size:
                combined = np.vstack([population, opp])
                combined_f = np.concatenate([fitness, opp_fitness])
                idx = np.argsort(combined_f)[:pop_size]
                population = combined[idx]
                fitness = combined_f[idx]

        mem_size = 8
        M_F = np.full(mem_size, 0.5)
        M_CR = np.full(mem_size, 0.85)
        k = 0
        archive = []
        gen = 0
        t_start = elapsed()
        total_time = time_limit - t_start

        while elapsed() < time_limit and pop_size >= N_min:
            S_F, S_CR, S_w = [], [], []
            gen += 1
            ratio = min(1.0, (elapsed() - t_start) / (total_time + 1e-30))
            new_pop_size = max(N_min, int(round(N_init - (N_init - N_min) * ratio)))

            trial_pop = np.empty_like(population)
            trial_fit = np.full(pop_size, float('inf'))

            for i in range(pop_size):
                if elapsed() >= time_limit:
                    break

                ri = np.random.randint(mem_size)
                Fi = -1
                while Fi <= 0:
                    Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                Fi = min(Fi, 1.0)

                CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
                if ratio > 0.7:
                    CRi = max(CRi, 0.5)

                p = max(2, int(max(0.05, 0.25 - 0.20 * ratio) * pop_size))
                pbest_idx = np.argsort(fitness)[:p]
                xpbest = population[np.random.choice(pbest_idx)]

                idxs = [j for j in range(pop_size) if j != i]
                a = np.random.choice(idxs)

                pool = list(range(pop_size))
                pool.remove(i)
                if a in pool:
                    pool.remove(a)
                if archive:
                    b_src = np.vstack([population[pool] if pool else population[[a]], np.array(archive)])
                    xb = b_src[np.random.randint(len(b_src))]
                else:
                    xb = population[np.random.choice(pool)] if pool else population[a]

                # current-to-pbest/1 with archive
                Fw = Fi * (0.5 + 0.5 * np.random.random())
                mutant = population[i] + Fi * (xpbest - population[i]) + Fw * (population[a] - xb)

                cross = np.random.rand(dim) < CRi
                cross[np.random.randint(dim)] = True
                trial = np.where(cross, mutant, population[i])

                for d in range(dim):
                    if trial[d] < lower[d]:
                        trial[d] = (lower[d] + population[i][d]) / 2
                    elif trial[d] > upper[d]:
                        trial[d] = (upper[d] + population[i][d]) / 2

                tf = evaluate(trial)

                if tf <= fitness[i]:
                    if tf < fitness[i]:
                        S_F.append(Fi)
                        S_CR.append(CRi)
                        S_w.append(abs(fitness[i] - tf))
                        archive.append(population[i].copy())
                        if len(archive) > N_init:
                            archive.pop(np.random.randint(len(archive)))
                    population[i] = trial
                    fitness[i] = tf

            if S_F:
                w = np.array(S_w)
                w /= w.sum() + 1e-30
                sf = np.array(S_F)
                M_F[k] = np.sum(w * sf ** 2) / (np.sum(w * sf) + 1e-30)
                M_CR[k] = np.sum(w * np.array(S_CR))
                k = (k + 1) % mem_size

            if new_pop_size < pop_size:
                idx = np.argsort(fitness)[:new_pop_size]
                population = population[idx]
                fitness = fitness[idx]
                pop_size = new_pop_size

    def cmaes_local(time_limit, x0, sigma0=0.05):
        nonlocal best, best_params
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu_v = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chiN = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        mean = x0.copy()
        sigma = sigma0 * np.mean(ranges)
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        gen = 0

        while elapsed() < time_limit:
            gen += 1
            try:
                D2, B = np.linalg.eigh(C)
                D2 = np.maximum(D2, 1e-20)
                D = np.sqrt(D2)
                invsqrtC = B @ np.diag(1.0 / D) @ B.T
            except:
                C = np.eye(n); D = np.ones(n); B = np.eye(n); invsqrtC = np.eye(n)

            arz = np.random.randn(lam, n)
            arx = np.array([clip(mean + sigma * (B @ (D * arz[k]))) for k in range(lam)])
            fitvals = []
            for k in range(lam):
                if elapsed() >= time_limit:
                    return
                fitvals.append(evaluate(arx[k]))
            fitvals = np.array(fitvals)

            idx = np.argsort(fitvals)
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mean - old_mean) / sigma
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * gen)) / chiN < 1.4 + 2 / (n + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma

            artmp = (arx[idx[:mu]] - old_mean) / sigma
            C = (1 - c1 - cmu_v) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_v * (artmp.T @ np.diag(weights) @ artmp)
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-16 * np.mean(ranges), np.mean(ranges))
            if sigma < 1e-14 * np.mean(ranges):
                break

    # Phase 1: L-SHADE (55% time)
    lshade_run(max_time * 0.55)

    # Phase 2: Restart L-SHADE from perturbed best solutions (20% time)
    if best_params is not None and elapsed() < max_time * 0.75:
        for restart_scale in [0.15, 0.05, 0.3]:
            if elapsed() >= max_time * 0.75:
                break
            n_pop = min(max(20, 5 * dim), 120)
            pop2 = np.array([best_params + np.random.normal(0, restart_scale, dim) * ranges for _ in range(n_pop)])
            pop2[0] = best_params.copy()
            pop2 = clip(pop2)
            lshade_run(min(elapsed() + (max_time * 0.75 - elapsed()) * 0.4, max_time * 0.75), pop_init=pop2)

    # Phase 3: CMA-ES local refinement (12% time)
    if best_params is not None and elapsed() < max_time * 0.87:
        cmaes_local(max_time * 0.87, best_params, sigma0=0.02)

    # Phase 4: Coordinate-wise golden section search (8% time)
    if best_params is not None and elapsed() < max_time * 0.95:
        x = best_params.copy()
        fx = best
        step = 0.01 * ranges
        for iteration in range(30):
            if elapsed() >= max_time * 0.95:
                break
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.95:
                    break
                for sign in [1, -1]:
                    xn = x.copy()
                    xn[d] = np.clip(x[d] + sign * step[d], lower[d], upper[d])
                    fn = evaluate(xn)
                    if fn < fx:
                        x, fx = xn, fn
                        improved = True
                        # Accelerate along this direction
                        for mult in [2, 4, 8]:
                            if elapsed() >= max_time * 0.95:
                                break
                            xn2 = x.copy()
                            xn2[d] = np.clip(x[d] + sign * step[d] * mult, lower[d], upper[d])
                            fn2 = evaluate(xn2)
                            if fn2 < fx:
                                x, fx = xn2, fn2
                            else:
                                break
                        break
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-13:
                    break

    # Phase 5: Final Nelder-Mead polish
    if best_params is not None and elapsed() < max_time * 0.995:
        n = dim
        scale = 0.002 * ranges
        simplex = np.zeros((n + 1, n))
        fs = np.zeros(n + 1)
        simplex[0] = best_params.copy()
        fs[0] = best
        for i in range(1, n + 1):
            if elapsed() >= max_time * 0.995:
                break
            simplex[i] = best_params.copy()
            simplex[i][(i - 1) % n] += scale[(i - 1) % n] * (1 if np.random.random() > 0.5 else -1)
            simplex[i] = clip(simplex[i])
            fs[i] = evaluate(simplex[i])
        for _ in range(10000):
            if elapsed() >= max_time * 0.995:
                break
            o = np.argsort(fs)
            simplex = simplex[o]
            fs = fs[o]
            c = np.mean(simplex[:-1], axis=0)
            xr = clip(c + (c - simplex[-1]))
            fr = evaluate(xr)
            if fr < fs[0]:
                xe = clip(c + 2 * (xr - c))
                fe = evaluate(xe)
                simplex[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < fs[-2]:
                simplex[-1] = xr
                fs[-1] = fr
            else:
                xc = clip(c + 0.5 * (simplex[-1] - c))
                fc = evaluate(xc)
                if fc < fs[-1]:
                    simplex[-1] = xc
                    fs[-1] = fc
                else:
                    for i in range(1, n + 1):
                        if elapsed() >= max_time * 0.995:
                            break
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = clip(simplex[i])
                        fs[i] = evaluate(simplex[i])
            if np.std(fs) < 1e-18:
                break

    return best
