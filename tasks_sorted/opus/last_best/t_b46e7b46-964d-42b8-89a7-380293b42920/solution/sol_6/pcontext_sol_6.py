#Looking at the previous algorithms, the best scores 3.19. I'll improve by:
#
#1. **Better L-SHADE with weighted mutation** and more careful archive management
#2. **Multiple restart phases** with decreasing search radii
#3. **Golden-section line search** for local refinement (more efficient than coordinate descent)
#4. **Adaptive Nelder-Mead** with restarts
#5. **Better time management** - more time for global search
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

    def lshade_run(time_end, pop_init=None):
        nonlocal best, best_params

        N_init = min(max(30, 8 * dim), 250)
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
            if elapsed() >= time_end:
                break
            fitness[i] = evaluate(population[i])

        # Opposition-based initialization
        if pop_init is None and elapsed() < time_end:
            opp = lower + upper - population
            opp_fitness = np.full(pop_size, float('inf'))
            for i in range(pop_size):
                if elapsed() >= time_end:
                    break
                opp_fitness[i] = evaluate(opp[i])
            if not np.any(np.isinf(opp_fitness)):
                combined = np.vstack([population, opp])
                combined_f = np.concatenate([fitness, opp_fitness])
                idx = np.argsort(combined_f)[:pop_size]
                population = combined[idx]
                fitness = combined_f[idx]

        mem_size = 10
        M_F = np.full(mem_size, 0.5)
        M_CR = np.full(mem_size, 0.85)
        k = 0
        archive = []
        t_start = elapsed()
        total_time = max(time_end - t_start, 1e-6)

        while elapsed() < time_end and pop_size >= N_min:
            S_F, S_CR, S_w = [], [], []
            ratio = min(1.0, (elapsed() - t_start) / total_time)
            new_pop_size = max(N_min, int(round(N_init - (N_init - N_min) * ratio)))

            sorted_idx = np.argsort(fitness)

            trial_pop = np.empty_like(population)
            trial_fit = np.full(pop_size, float('inf'))
            
            Fs = np.zeros(pop_size)
            CRs = np.zeros(pop_size)
            
            for i in range(pop_size):
                ri = np.random.randint(mem_size)
                Fi = -1
                while Fi <= 0:
                    Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                Fi = min(Fi, 1.0)
                Fs[i] = Fi
                CRs[i] = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)

            for i in range(pop_size):
                if elapsed() >= time_end:
                    break

                Fi = Fs[i]
                CRi = CRs[i]

                p = max(2, int(max(0.05, 0.25 - 0.20 * ratio) * pop_size))
                xpbest = population[sorted_idx[np.random.randint(p)]]

                idxs = list(range(pop_size))
                idxs.remove(i)
                a = np.random.choice(idxs)
                idxs.remove(a)

                if archive and np.random.random() < 0.5:
                    xb = archive[np.random.randint(len(archive))]
                elif idxs:
                    xb = population[np.random.choice(idxs)]
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

    # Phase 1: Main L-SHADE (60%)
    lshade_run(max_time * 0.60)

    # Phase 2: Restart with perturbation around best
    for scale in [0.15, 0.05, 0.01]:
        if best_params is None or elapsed() >= max_time * 0.82:
            break
        n_pop = min(max(20, 5 * dim), 120)
        pop2 = np.array([best_params + np.random.normal(0, scale, dim) * ranges for _ in range(n_pop)])
        pop2[0] = best_params.copy()
        pop2 = clip(pop2)
        t_alloc = min(elapsed() + (max_time * 0.82 - elapsed()) * 0.45, max_time * 0.82)
        lshade_run(t_alloc, pop_init=pop2)

    # Phase 3: Coordinate descent with golden section
    if best_params is not None and elapsed() < max_time * 0.94:
        x = best_params.copy()
        fx = best
        step = 0.01 * ranges
        for _ in range(30):
            if elapsed() >= max_time * 0.94: break
            improved = False
            for d in np.random.permutation(dim):
                if elapsed() >= max_time * 0.94: break
                for sign in [1, -1]:
                    xn = x.copy()
                    xn[d] = np.clip(x[d] + sign * step[d], lower[d], upper[d])
                    fn = evaluate(xn)
                    if fn < fx:
                        x, fx = xn, fn
                        improved = True
                        for mult in [2, 4, 8]:
                            if elapsed() >= max_time * 0.94: break
                            xn2 = x.copy()
                            xn2[d] = np.clip(x[d] + sign * step[d] * mult, lower[d], upper[d])
                            fn2 = evaluate(xn2)
                            if fn2 < fx: x, fx = xn2, fn2
                            else: break
                        break
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-13: break

    # Phase 4: Nelder-Mead polish
    if best_params is not None and elapsed() < max_time * 0.998:
        n = dim
        scale = 0.002 * ranges
        simplex = np.zeros((n + 1, n))
        fs = np.zeros(n + 1)
        simplex[0] = best_params.copy()
        fs[0] = best
        for i in range(1, n + 1):
            if elapsed() >= max_time * 0.998: break
            simplex[i] = best_params.copy()
            simplex[i][(i - 1) % n] += scale[(i - 1) % n] * (1 if np.random.random() > 0.5 else -1)
            simplex[i] = clip(simplex[i])
            fs[i] = evaluate(simplex[i])
        for _ in range(10000):
            if elapsed() >= max_time * 0.998: break
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
                        if elapsed() >= max_time * 0.998: break
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = clip(simplex[i]); fs[i] = evaluate(simplex[i])
            if np.std(fs) < 1e-18: break

    return best
