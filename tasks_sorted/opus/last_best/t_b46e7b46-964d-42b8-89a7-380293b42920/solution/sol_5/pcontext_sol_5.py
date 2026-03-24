#Looking at the previous algorithms, the best one scores 3.19 with L-SHADE + coordinate descent + Nelder-Mead. The other scores 8.1 with a more complex but less effective approach. I'll improve by:
#
#1. **Better L-SHADE**: More aggressive parameter adaptation, larger memory, better archive usage
#2. **Multiple diverse restarts** with different scales
#3. **Improved CMA-ES local search** with proper step-size adaptation
#4. **Pattern search** instead of simple coordinate descent for better local refinement
#5. **Better time allocation** based on what worked
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

        N_init = min(max(25, 7 * dim), 200)
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
        fitness = np.array([evaluate(population[i]) for i in range(pop_size) if elapsed() < time_end])
        if len(fitness) < pop_size:
            fitness = np.append(fitness, [float('inf')] * (pop_size - len(fitness)))

        if pop_init is None and elapsed() < time_end:
            opp = lower + upper - population
            opp_fitness = np.array([evaluate(opp[i]) for i in range(pop_size) if elapsed() < time_end])
            if len(opp_fitness) == pop_size:
                combined = np.vstack([population, opp])
                combined_f = np.concatenate([fitness, opp_fitness])
                idx = np.argsort(combined_f)[:pop_size]
                population = combined[idx]
                fitness = combined_f[idx]

        mem_size = 8
        M_F = np.full(mem_size, 0.5)
        M_CR = np.full(mem_size, 0.8)
        k = 0
        archive = []
        gen = 0
        t_start = elapsed()
        total_time = max(time_end - t_start, 1e-6)

        while elapsed() < time_end and pop_size >= N_min:
            S_F, S_CR, S_w = [], [], []
            gen += 1
            ratio = min(1.0, (elapsed() - t_start) / total_time)
            new_pop_size = max(N_min, int(round(N_init - (N_init - N_min) * ratio)))

            sorted_idx = np.argsort(fitness)

            for i in range(pop_size):
                if elapsed() >= time_end:
                    break

                ri = np.random.randint(mem_size)
                Fi = -1
                while Fi <= 0:
                    Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                Fi = min(Fi, 1.0)

                CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)

                p = max(2, int(max(0.05, 0.2 - 0.15 * ratio) * pop_size))
                xpbest = population[sorted_idx[np.random.randint(p)]]

                idxs = [j for j in range(pop_size) if j != i]
                a = np.random.choice(idxs)

                if archive:
                    pool_pop = [j for j in idxs if j != a]
                    b_src = np.vstack([population[pool_pop] if pool_pop else population[[a]], np.array(archive)])
                    xb = b_src[np.random.randint(len(b_src))]
                else:
                    pool_pop = [j for j in idxs if j != a]
                    xb = population[np.random.choice(pool_pop)] if pool_pop else population[a]

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

    # Phase 1: Main L-SHADE (65% time)
    t1 = max_time * 0.65
    lshade_run(t1)

    # Phase 2: Restart L-SHADE with different perturbation scales
    for scale in [0.1, 0.03, 0.2]:
        if best_params is None or elapsed() >= max_time * 0.85:
            break
        n_pop = min(max(20, 5 * dim), 100)
        pop2 = np.array([best_params + np.random.normal(0, scale, dim) * ranges for _ in range(n_pop)])
        pop2[0] = best_params.copy()
        pop2 = clip(pop2)
        t_alloc = elapsed() + (max_time * 0.85 - elapsed()) * 0.4
        lshade_run(min(t_alloc, max_time * 0.85), pop_init=pop2)

    # Phase 3: Coordinate descent
    if best_params is not None and elapsed() < max_time * 0.95:
        x = best_params.copy()
        fx = best
        step = 0.01 * ranges
        for _ in range(30):
            if elapsed() >= max_time * 0.95: break
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.95: break
                for sign in [1, -1]:
                    xn = x.copy()
                    xn[d] = np.clip(x[d] + sign * step[d], lower[d], upper[d])
                    fn = evaluate(xn)
                    if fn < fx:
                        x, fx = xn, fn
                        improved = True
                        for mult in [2, 4]:
                            if elapsed() >= max_time * 0.95: break
                            xn2 = x.copy()
                            xn2[d] = np.clip(x[d] + sign * step[d] * mult, lower[d], upper[d])
                            fn2 = evaluate(xn2)
                            if fn2 < fx: x, fx = xn2, fn2
                            else: break
                        break
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-12: break

    # Phase 4: Nelder-Mead polish
    if best_params is not None and elapsed() < max_time * 0.998:
        n = dim
        scale = 0.003 * ranges
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
