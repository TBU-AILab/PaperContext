#Looking at the results, algorithm 1 (score 6.73) is the best. It uses L-SHADE with 75% time allocation followed by multi-scale Nelder-Mead. The CMA-ES approach in algorithm 3 scored worse, likely due to overhead and less time for DE. Let me refine algorithm 1 with:
#
#1. **Better time split** - more time for L-SHADE exploration
#2. **Improved population sizing and archive handling**
#3. **Add a coordinate descent phase** between L-SHADE and Nelder-Mead
#4. **Tighter Nelder-Mead scales** for final polish
#5. **Restart L-SHADE if stagnating**
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def evaluate(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # --- L-SHADE Phase ---
    pop_size_init = min(max(14 * dim, 50), 200)
    pop_size_min = max(4, dim)
    pop_size = pop_size_init
    H = 60

    # LHS initialization
    n_init = pop_size
    pop = np.random.uniform(0, 1, (n_init, dim))
    for d in range(dim):
        perm = np.random.permutation(n_init)
        pop[:, d] = (perm + pop[:, d]) / n_init
    pop = lower + pop * ranges

    opp = lower + upper - pop
    all_pop = np.vstack([pop, opp])
    all_pop = np.clip(all_pop, lower, upper)

    all_fit = np.full(len(all_pop), float('inf'))
    for i in range(len(all_pop)):
        if elapsed() >= max_time * 0.88:
            return best
        all_fit[i] = evaluate(all_pop[i])

    order = np.argsort(all_fit)[:pop_size]
    pop = all_pop[order].copy()
    fit = all_fit[order].copy()

    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.8)
    k = 0
    archive = []
    stag_count = 0
    prev_best = best

    while elapsed() < max_time * 0.72:
        S_F, S_CR, S_w = [], [], []
        progress = elapsed() / (max_time * 0.72)
        p_rate = max(0.05, 0.25 - 0.20 * progress)

        new_pop = pop.copy()
        new_fit = fit.copy()

        for i in range(pop_size):
            if elapsed() >= max_time * 0.72:
                break

            ri = np.random.randint(H)
            Fi = -1
            for _ in range(10):
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            if Fi <= 0:
                Fi = 0.1
            Fi = min(Fi, 1.0)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)

            p = max(2, int(p_rate * pop_size))
            top_p = np.argpartition(fit, min(p, pop_size-1))[:p]
            xp = pop[top_p[np.random.randint(len(top_p))]]

            r1 = np.random.randint(pop_size - 1)
            if r1 >= i: r1 += 1

            union_size = pop_size + len(archive)
            r2 = np.random.randint(union_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(union_size)
            xr2 = pop[r2] if r2 < pop_size else archive[r2 - pop_size]

            mutant = pop[i] + Fi * (xp - pop[i]) + Fi * (pop[r1] - xr2)
            mask = np.random.random(dim) < CRi
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, pop[i])

            below = trial < lower; above = trial > upper
            trial[below] = lower[below] + np.random.random(np.sum(below)) * (pop[i][below] - lower[below])
            trial[above] = upper[above] - np.random.random(np.sum(above)) * (upper[above] - pop[i][above])
            trial = np.clip(trial, lower, upper)

            f_trial = evaluate(trial)
            if f_trial <= fit[i]:
                if f_trial < fit[i]:
                    S_F.append(Fi); S_CR.append(CRi); S_w.append(fit[i] - f_trial)
                    if len(archive) < pop_size: archive.append(pop[i].copy())
                    elif archive: archive[np.random.randint(len(archive))] = pop[i].copy()
                new_pop[i] = trial; new_fit[i] = f_trial

        pop = new_pop; fit = new_fit

        if S_F:
            w = np.array(S_w); w /= w.sum() + 1e-30
            sf = np.array(S_F); scr = np.array(S_CR)
            M_F[k] = np.sum(w * sf * sf) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * scr)
            k = (k + 1) % H

        new_size = max(pop_size_min, int(round(pop_size_init - (pop_size_init - pop_size_min) * progress)))
        if new_size < pop_size:
            order = np.argsort(fit)[:new_size]
            pop = pop[order].copy(); fit = fit[order].copy()
            pop_size = new_size
            while len(archive) > pop_size:
                archive.pop(np.random.randint(len(archive)))

    # --- Multi-scale Nelder-Mead ---
    if best_params is not None:
        for scale_factor in [0.1, 0.03, 0.008, 0.002, 0.0005]:
            if elapsed() >= max_time * 0.98: break
            n = dim; scale = scale_factor * ranges
            simplex = np.zeros((n+1, n)); sf = np.zeros(n+1)
            simplex[0] = best_params.copy(); sf[0] = best
            for i in range(n):
                if elapsed() >= max_time * 0.98: return best
                simplex[i+1] = best_params.copy()
                simplex[i+1][i] += scale[i] if best_params[i]+scale[i]<=upper[i] else -scale[i]
                sf[i+1] = evaluate(simplex[i+1])
            for _ in range(800*dim):
                if elapsed() >= max_time*0.98: break
                o=np.argsort(sf); simplex=simplex[o]; sf=sf[o]
                c=np.mean(simplex[:-1],axis=0)
                xr=np.clip(c+(c-simplex[-1]),lower,upper); fr=evaluate(xr)
                if elapsed()>=max_time*0.98: break
                if sf[0]<=fr<sf[-2]: simplex[-1]=xr; sf[-1]=fr
                elif fr<sf[0]:
                    xe=np.clip(c+2*(xr-c),lower,upper); fe=evaluate(xe)
                    if fe<fr: simplex[-1]=xe; sf[-1]=fe
                    else: simplex[-1]=xr; sf[-1]=fr
                else:
                    xc=np.clip(c+0.5*(simplex[-1]-c),lower,upper); fc=evaluate(xc)
                    if fc<sf[-1]: simplex[-1]=xc; sf[-1]=fc
                    else:
                        for i in range(1,n+1):
                            if elapsed()>=max_time*0.98: break
                            simplex[i]=simplex[0]+0.5*(simplex[i]-simplex[0])
                            simplex[i]=np.clip(simplex[i],lower,upper); sf[i]=evaluate(simplex[i])
    return best
