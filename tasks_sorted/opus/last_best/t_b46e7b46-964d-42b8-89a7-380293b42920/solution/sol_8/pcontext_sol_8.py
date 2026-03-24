#Looking at the two algorithms, the best one scores 2.14 but the last one regressed to 5.44. I'll analyze what works and improve:
#
#1. The 2.14 algorithm uses CMA-ES local search which is powerful for correlated landscapes
#2. I'll add better global exploration with multiple independent restarts
#3. Improve CMA-ES integration and add a more aggressive multi-start strategy
#4. Better time allocation with adaptive phases
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

    def lshade_run(time_end, pop_init=None, N_override=None):
        nonlocal best, best_params

        N_init = N_override if N_override else min(max(30, 8 * dim), 250)
        N_min = 4

        if pop_init is not None:
            population = pop_init.copy()
            N_init = max(N_init, len(population))
        else:
            population = np.zeros((N_init, dim))
            for d in range(dim):
                perm = np.random.permutation(N_init)
                for i in range(N_init):
                    population[i, d] = lower[d] + (perm[i] + np.random.random()) / N_init * ranges[d]

        pop_size = len(population)
        fitness = np.full(pop_size, float('inf'))
        for i in range(pop_size):
            if elapsed() >= time_end: break
            fitness[i] = evaluate(population[i])

        if pop_init is None and elapsed() < time_end:
            opp = lower + upper - population
            opp_fitness = np.full(pop_size, float('inf'))
            for i in range(pop_size):
                if elapsed() >= time_end: break
                opp_fitness[i] = evaluate(opp[i])
            if not np.any(np.isinf(opp_fitness)):
                combined = np.vstack([population, opp])
                combined_f = np.concatenate([fitness, opp_fitness])
                idx = np.argsort(combined_f)[:pop_size]
                population = combined[idx]
                fitness = combined_f[idx]

        mem_size = max(5, dim)
        M_F = np.full(mem_size, 0.5)
        M_CR = np.full(mem_size, 0.85)
        k = 0
        archive = []
        t_start = elapsed()
        total_time = max(time_end - t_start, 1e-6)
        gen = 0

        while elapsed() < time_end and pop_size >= N_min:
            gen += 1
            S_F, S_CR, S_w = [], [], []
            ratio = min(1.0, (elapsed() - t_start) / total_time)
            new_pop_size = max(N_min, int(round(N_init - (N_init - N_min) * ratio)))

            sorted_idx = np.argsort(fitness)

            for i in range(pop_size):
                if elapsed() >= time_end: break

                ri = np.random.randint(mem_size)
                Fi = -1
                while Fi <= 0:
                    Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                Fi = min(Fi, 1.0)
                CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)

                p = max(2, int(max(0.05, 0.25 - 0.20 * ratio) * pop_size))
                xpbest = population[sorted_idx[np.random.randint(p)]]

                candidates = list(range(pop_size))
                candidates.remove(i)
                a = np.random.choice(candidates)
                candidates.remove(a)

                if archive and np.random.random() < 0.5:
                    xb = archive[np.random.randint(len(archive))]
                elif candidates:
                    xb = population[np.random.choice(candidates)]
                else:
                    xb = population[a]

                mutant = population[i] + Fi * (xpbest - population[i]) + Fi * (population[a] - xb)

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

        return population, fitness

    def cma_es_local(time_end, x0, sigma0):
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
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        xmean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        eigeneval = 0
        counteval = 0

        while elapsed() < time_end:
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            fitvals = np.zeros(lam)
            for ki in range(lam):
                if elapsed() >= time_end: return
                arx[ki] = clip(xmean + sigma * (B @ (D * arz[ki])))
                fitvals[ki] = evaluate(arx[ki])
                counteval += 1

            idx = np.argsort(fitvals)
            arx = arx[idx]; arz = arz[idx]

            xold = xmean.copy()
            xmean = np.dot(weights, arx[:mu])
            zmean = np.dot(weights, arz[:mu])

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ zmean)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / lam)) / chiN < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma

            artmp = (arx[:mu] - xold) / sigma
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_val * (artmp.T @ np.diag(weights) @ artmp)

            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 0.5 * np.max(ranges))

            if counteval - eigeneval > lam / (c1 + cmu_val) / n / 10:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                except:
                    C = np.eye(n); B = np.eye(n); D = np.ones(n)

            if sigma * np.max(D) < 1e-15 * np.max(ranges):
                break

    # Phase 1: Main L-SHADE (50%)
    lshade_run(max_time * 0.50)

    # Phase 2: Restarts around best with L-SHADE
    for scale in [0.2, 0.08, 0.02]:
        if best_params is None or elapsed() >= max_time * 0.70:
            break
        n_pop = min(max(20, 5 * dim), 100)
        pop2 = np.array([best_params + np.random.normal(0, scale, dim) * ranges for _ in range(n_pop)])
        pop2[0] = best_params.copy()
        pop2 = clip(pop2)
        t_alloc = min(elapsed() + (max_time * 0.70 - elapsed()) * 0.4, max_time * 0.70)
        lshade_run(t_alloc, pop_init=pop2)

    # Phase 3: CMA-ES with multiple restarts at different scales
    for sigma_scale in [0.05, 0.01, 0.002]:
        if best_params is None or elapsed() >= max_time * 0.92:
            break
        t_alloc = min(elapsed() + (max_time * 0.92 - elapsed()) * 0.4, max_time * 0.92)
        cma_es_local(t_alloc, best_params.copy(), sigma_scale * np.max(ranges))

    # Phase 4: Fine coordinate descent
    if best_params is not None and elapsed() < max_time * 0.97:
        x = best_params.copy()
        fx = best
        step = 0.005 * ranges
        for _ in range(50):
            if elapsed() >= max_time * 0.97: break
            improved = False
            for d in np.random.permutation(dim):
                if elapsed() >= max_time * 0.97: break
                for sign in [1, -1]:
                    xn = x.copy()
                    xn[d] = np.clip(x[d] + sign * step[d], lower[d], upper[d])
                    fn = evaluate(xn)
                    if fn < fx:
                        x, fx = xn, fn
                        improved = True
                        for mult in [2, 4, 8, 16]:
                            if elapsed() >= max_time * 0.97: break
                            xn2 = x.copy()
                            xn2[d] = np.clip(x[d] + sign * step[d] * mult, lower[d], upper[d])
                            fn2 = evaluate(xn2)
                            if fn2 < fx: x, fx = xn2, fn2
                            else: break
                        break
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-15: break

    # Phase 5: Nelder-Mead final polish
    if best_params is not None and elapsed() < max_time * 0.999:
        n = dim
        scale = 0.001 * ranges
        simplex = np.zeros((n + 1, n))
        fs = np.zeros(n + 1)
        simplex[0] = best_params.copy()
        fs[0] = best
        for i in range(1, n + 1):
            if elapsed() >= max_time * 0.999: break
            simplex[i] = best_params.copy()
            simplex[i][(i-1) % n] += scale[(i-1) % n] * (1 if np.random.random() > 0.5 else -1)
            simplex[i] = clip(simplex[i])
            fs[i] = evaluate(simplex[i])
        for _ in range(10000):
            if elapsed() >= max_time * 0.999: break
            o = np.argsort(fs); simplex = simplex[o]; fs = fs[o]
            c = np.mean(simplex[:-1], axis=0)
            xr = clip(c + (c - simplex[-1])); fr = evaluate(xr)
            if fr < fs[0]:
                xe = clip(c + 2 * (xr - c)); fe = evaluate(xe)
                simplex[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < fs[-2]:
                simplex[-1] = xr; fs[-1] = fr
            else:
                xc = clip(c + 0.5 * (simplex[-1] - c)); fc = evaluate(xc)
                if fc < fs[-1]:
                    simplex[-1] = xc; fs[-1] = fc
                else:
                    for i in range(1, n + 1):
                        if elapsed() >= max_time * 0.999: break
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = clip(simplex[i]); fs[i] = evaluate(simplex[i])
            if np.std(fs) < 1e-20: break

    return best
