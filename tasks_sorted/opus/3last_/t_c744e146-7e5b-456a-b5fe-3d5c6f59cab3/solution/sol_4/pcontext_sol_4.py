#Looking at the results, algorithm 1 performed best (6.025). I'll improve upon it by:
#
#1. Better L-SHADE with larger population and weighted recombination
#2. Adding a Nelder-Mead simplex local search phase
#3. Better CMA-ES with IPOP restarts (increasing population)
#4. More aggressive local search at the end
#5. Better time budget management
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
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_f(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    def lhs_sample(n):
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            for i in range(n):
                samples[i, d] = lower[d] + (perm[i] + np.random.rand()) / n * ranges[d]
        return samples

    # ---- CMA-ES ----
    def cma_es(x0, sigma0, max_t=None, lam_mult=1):
        nonlocal best, best_x
        n = dim
        lam = int((4 + int(3 * np.log(n))) * lam_mult)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        
        xmean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        invsqrtC = np.eye(n)
        eigeneval = 0
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        evals = 0
        t_end = elapsed() + max_t if max_t else float('inf')
        no_improve = 0
        prev_best_local = best
        
        while elapsed() < t_end and time_left() > 0.05:
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = xmean + sigma * (B @ (D * arz[k]))
                arx[k] = clip(arx[k])
            
            fit = np.array([eval_f(arx[k]) for k in range(lam)])
            evals += lam
            
            idx = np.argsort(fit)
            arx = arx[idx]
            arz = arz[idx]
            
            xold = xmean.copy()
            xmean = arx[:mu].T @ weights
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ ((xmean - xold) / sigma))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*evals/lam)) / chiN) < (1.4 + 2/(n+1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ((xmean - xold) / sigma)
            
            artmp = (arx[:mu] - xold) / sigma
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1-hsig)*cc*(2-cc)*C) + cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, max(ranges))
            
            if evals - eigeneval > lam / (c1 + cmu_val) / n / 10:
                eigeneval = evals
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0/D) @ B.T
                except:
                    break
            
            if best < prev_best_local - 1e-10:
                no_improve = 0
                prev_best_local = best
            else:
                no_improve += 1
            
            if sigma * np.max(D) < 1e-14 * max(ranges) or no_improve > 40 + 8*n:
                break

    # ---- Nelder-Mead ----
    def nelder_mead(x0, max_t=None, initial_scale=0.05):
        nonlocal best, best_x
        n = dim
        t_end = elapsed() + max_t if max_t else float('inf')
        
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += initial_scale * ranges[i]
            simplex[i + 1] = clip(simplex[i + 1])
        
        f_simplex = np.array([eval_f(simplex[i]) for i in range(n + 1)])
        
        while elapsed() < t_end and time_left() > 0.05:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = simplex[:n].mean(axis=0)
            
            xr = clip(centroid + alpha * (centroid - simplex[n]))
            fr = eval_f(xr)
            
            if fr < f_simplex[0]:
                xe = clip(centroid + gamma * (xr - centroid))
                fe = eval_f(xe)
                if fe < fr:
                    simplex[n] = xe; f_simplex[n] = fe
                else:
                    simplex[n] = xr; f_simplex[n] = fr
            elif fr < f_simplex[n - 1]:
                simplex[n] = xr; f_simplex[n] = fr
            else:
                xc = clip(centroid + rho * (simplex[n] - centroid))
                fc = eval_f(xc)
                if fc < f_simplex[n]:
                    simplex[n] = xc; f_simplex[n] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                        f_simplex[i] = eval_f(simplex[i])
            
            if np.max(np.abs(simplex[n] - simplex[0])) < 1e-15:
                break

    # ---- L-SHADE (50% time) ----
    pop_size_init = min(max(60, 18 * dim), 600)
    pop_size = pop_size_init
    min_pop = 4
    population = lhs_sample(pop_size)
    fitness = np.array([eval_f(ind) for ind in population])
    
    memory_size = 10
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.8)
    mem_idx = 0
    archive = []
    archive_max = pop_size_init
    shade_end = max_time * 0.50
    
    while elapsed() < shade_end and time_left() > 0.5:
        S_F, S_CR, delta_f = [], [], []
        p_best_rate = max(2.0/pop_size, 0.05)
        for i in range(pop_size):
            if elapsed() >= shade_end or time_left() < 0.3:
                break
            ri = np.random.randint(0, memory_size)
            Fi = -1
            while Fi <= 0:
                Fi = min(M_F[ri] + 0.1 * np.random.standard_cauchy(), 1.0)
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            p = max(2, int(p_best_rate * pop_size))
            sorted_idx = np.argsort(fitness)
            pbest = sorted_idx[np.random.randint(0, p)]
            cands = [j for j in range(pop_size) if j != i]
            r1 = cands[np.random.randint(len(cands))]
            pool = [j for j in range(pop_size) if j != i and j != r1]
            r2c = np.random.randint(len(pool) + len(archive))
            xr2 = population[pool[r2c]] if r2c < len(pool) else archive[r2c - len(pool)]
            mutant = population[i] + Fi * (population[pbest] - population[i]) + Fi * (population[r1] - xr2)
            cross = np.random.rand(dim) < CRi
            cross[np.random.randint(dim)] = True
            trial = np.where(cross, mutant, population[i])
            for d in range(dim):
                if trial[d] < lower[d]: trial[d] = (lower[d] + population[i][d]) / 2
                elif trial[d] > upper[d]: trial[d] = (upper[d] + population[i][d]) / 2
            tf = eval_f(trial)
            if tf <= fitness[i]:
                if tf < fitness[i]:
                    S_F.append(Fi); S_CR.append(CRi); delta_f.append(fitness[i] - tf)
                    archive.append(population[i].copy())
                    if len(archive) > archive_max: archive.pop(np.random.randint(len(archive)))
                population[i] = trial; fitness[i] = tf
        if S_F:
            w = np.array(delta_f); w /= w.sum() + 1e-30
            M_F[mem_idx] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[mem_idx] = np.sum(w * np.array(S_CR))
            mem_idx = (mem_idx + 1) % memory_size
        new_ps = max(min_pop, int(pop_size_init - (pop_size_init - min_pop) * elapsed() / shade_end))
        if new_ps < pop_size:
            sidx = np.argsort(fitness)
            population = population[sidx[:new_ps]]; fitness = fitness[sidx[:new_ps]]; pop_size = new_ps

    # ---- IPOP-CMA-ES restarts ----
    restart = 0
    lam_mult = 1.0
    while time_left() > 0.5:
        restart += 1
        if restart == 1:
            x0 = best_x.copy(); s0 = np.mean(ranges) * 0.2
        elif restart == 2:
            x0 = best_x.copy(); s0 = np.mean(ranges) * 0.02; lam_mult = 1.0
        else:
            lam_mult = min(lam_mult * 2, 8)
            x0 = best_x + np.random.randn(dim) * ranges * 0.15
            x0 = clip(x0); s0 = np.mean(ranges) * 0.15
        cma_es(x0, s0, max_t=max(time_left() * 0.35, 0.3), lam_mult=lam_mult)

    # ---- Nelder-Mead polish ----
    if best_x is not None and time_left() > 0.3:
        nelder_mead(best_x.copy(), max_t=time_left() * 0.8, initial_scale=0.01)

    # ---- Fine coordinate descent ----
    if best_x is not None and time_left() > 0.1:
        x_local = best_x.copy()
        f_local = best
        step = ranges * 0.0005
        for _ in range(20):
            if time_left() < 0.03: break
            improved = False
            for d in range(dim):
                if time_left() < 0.02: break
                for sign in [1, -1]:
                    xt = x_local.copy()
                    xt[d] = np.clip(xt[d] + sign * step[d], lower[d], upper[d])
                    ft = eval_f(xt)
                    if ft < f_local:
                        f_local = ft; x_local = xt; improved = True; break
            if not improved: step *= 0.5

    return best
