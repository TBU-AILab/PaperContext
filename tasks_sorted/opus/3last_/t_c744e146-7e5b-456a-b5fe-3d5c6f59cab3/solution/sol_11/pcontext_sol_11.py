#Looking at the results, algorithm 1 (L-SHADE + CMA-ES + Pattern search) performed best with 22.598. I'll improve it by:
#
#1. Better initial sampling (Sobol-like)
#2. More aggressive L-SHADE with improved memory and population sizing
#3. Multiple CMA-ES restarts with IPOP strategy
#4. Nelder-Mead simplex refinement instead of simple pattern search
#5. Better time allocation
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
    evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time - elapsed()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_f(x):
        nonlocal best, best_x, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # ---- L-SHADE Phase (45% of time) ----
    pop_size_init = min(max(80, 20 * dim), 800)
    pop_size = pop_size_init
    min_pop = 4
    
    # LHS initialization
    population = np.zeros((pop_size_init, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size_init)
        for i in range(pop_size_init):
            population[i, d] = lower[d] + (perm[i] + np.random.rand()) / pop_size_init * ranges[d]
    
    fitness = np.array([eval_f(ind) for ind in population])
    
    # Opposition-based learning
    if time_left() > max_time * 0.8:
        opp = lower + upper - population
        opp_fit = np.array([eval_f(ind) for ind in opp])
        all_pop = np.vstack([population, opp])
        all_fit = np.concatenate([fitness, opp_fit])
        idx = np.argsort(all_fit)[:pop_size_init]
        population = all_pop[idx]
        fitness = all_fit[idx]
    
    memory_size = 12
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.8)
    mem_idx = 0
    archive = []
    archive_max = int(pop_size_init * 1.5)
    shade_end = max_time * 0.45
    gen = 0
    max_gen_est = max(1, int(shade_end * pop_size_init / (pop_size_init * 0.001 + dim * 0.01) / pop_size_init))
    
    while elapsed() < shade_end and time_left() > 0.5:
        gen += 1
        S_F, S_CR, delta_f = [], [], []
        
        progress = elapsed() / shade_end
        p_best_rate = max(2.0 / pop_size, 0.05 + 0.15 * (1 - progress))
        
        trial_pop = np.empty_like(population[:pop_size])
        trial_fitness = np.full(pop_size, np.inf)
        
        for i in range(pop_size):
            if elapsed() >= shade_end or time_left() < 0.3:
                trial_pop[i] = population[i]
                trial_fitness[i] = fitness[i]
                continue
            
            ri = np.random.randint(0, memory_size)
            Fi = -1
            while Fi <= 0:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi >= 1: 
                    Fi = 1.0
                    break
                if Fi <= 0:
                    Fi = -1
            
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            # For low-dimensional problems, occasionally use low CR
            if dim <= 10 and np.random.rand() < 0.1:
                CRi = 0.0
            
            p = max(2, int(p_best_rate * pop_size))
            sorted_idx = np.argsort(fitness[:pop_size])
            pbest = sorted_idx[np.random.randint(0, p)]
            
            cands = list(range(pop_size))
            cands.remove(i)
            r1 = cands[np.random.randint(len(cands))]
            
            pool2 = [j for j in range(pop_size) if j != i and j != r1]
            total_pool = len(pool2) + len(archive)
            if total_pool == 0:
                xr2 = population[r1]
            else:
                r2c = np.random.randint(total_pool)
                if r2c < len(pool2):
                    xr2 = population[pool2[r2c]]
                else:
                    xr2 = archive[r2c - len(pool2)]
            
            mutant = population[i] + Fi * (population[pbest] - population[i]) + Fi * (population[r1] - xr2)
            
            cross = np.random.rand(dim) < CRi
            cross[np.random.randint(dim)] = True
            trial = np.where(cross, mutant, population[i])
            
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + population[i][d]) / 2
                elif trial[d] > upper[d]:
                    trial[d] = (upper[d] + population[i][d]) / 2
            
            tf = eval_f(trial)
            trial_pop[i] = trial
            trial_fitness[i] = tf
            
            if tf <= fitness[i]:
                if tf < fitness[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    delta_f.append(abs(fitness[i] - tf))
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(len(archive)))
                population[i] = trial
                fitness[i] = tf
            # else keep original (already in population)
        
        if S_F:
            w = np.array(delta_f)
            ws = w.sum()
            if ws > 0:
                w = w / ws
            else:
                w = np.ones(len(w)) / len(w)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[mem_idx] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[mem_idx] = np.sum(w * scr)
            mem_idx = (mem_idx + 1) % memory_size
        
        # Linear population size reduction
        new_ps = max(min_pop, int(round(pop_size_init - (pop_size_init - min_pop) * progress)))
        if new_ps < pop_size:
            sidx = np.argsort(fitness[:pop_size])
            population = population[sidx[:new_ps]]
            fitness = fitness[sidx[:new_ps]]
            pop_size = new_ps

    # ---- CMA-ES Phase with IPOP restarts (45% of time) ----
    def cma_es(x0, sigma0, max_t, lam_mult=1):
        nonlocal best, best_x
        n = dim
        lam = int((4 + int(3 * np.log(n))) * lam_mult)
        if lam < 6: lam = 6
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_v = min(1 - c1, 2*(mueff - 2 + 1/mueff)/((n+2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1))-1) + cs
        chiN = n**0.5*(1-1/(4*n)+1/(21*n**2))
        xmean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        invsqrtC = np.eye(n)
        ev = 0
        eigeneval = 0
        t_end = elapsed() + max_t
        best_local = float('inf')
        stag = 0
        
        while elapsed() < t_end and time_left() > 0.05:
            arx = np.zeros((lam, n))
            for k in range(lam):
                arx[k] = clip(xmean + sigma * (B @ (D * np.random.randn(n))))
            fit = np.array([eval_f(arx[k]) for k in range(lam)])
            ev += lam
            
            idx = np.argsort(fit)
            if fit[idx[0]] < best_local:
                best_local = fit[idx[0]]
                stag = 0
            else:
                stag += 1
            
            xold = xmean.copy()
            xmean = arx[idx[:mu]].T @ weights
            
            diff = (xmean - xold) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ diff)
            hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * ev / lam)) / chiN < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            artmp = (arx[idx[:mu]] - xold) / sigma
            C = (1 - c1 - cmu_v) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu_v * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            
            if ev - eigeneval > lam / (c1 + cmu_v + 1e-20) / n / 10:
                eigeneval = ev
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D2, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D2, 1e-20))
                    invsqrtC = B @ np.diag(1 / D) @ B.T
                except:
                    break
            
            if sigma * np.max(D) < 1e-14 * np.max(ranges):
                break
            if stag > 20 + dim:
                break
            # Condition number check
            if np.max(D) / (np.min(D) + 1e-30) > 1e14:
                break

    # IPOP-CMA-ES: increasing population restarts
    cma_time = max_time * 0.45
    lam_mult = 1.0
    restart = 0
    
    while time_left() > 1.0 and elapsed() < max_time * 0.90:
        restart += 1
        # Start from best + perturbation or random
        if restart == 1:
            x0 = best_x.copy()
            sig = np.mean(ranges) * 0.15
        elif restart <= 3:
            x0 = best_x.copy() + np.random.randn(dim) * ranges * 0.05
            x0 = clip(x0)
            sig = np.mean(ranges) * (0.05 / restart)
        else:
            x0 = lower + np.random.rand(dim) * ranges
            sig = np.mean(ranges) * 0.25
            lam_mult = min(lam_mult * 2, 10)
        
        alloc = min(time_left() * 0.4, time_left() - 0.5)
        if alloc < 0.3:
            break
        cma_es(x0, sig, alloc, lam_mult)

    # ---- Nelder-Mead simplex polish (remaining time) ----
    if best_x is not None and time_left() > 0.1:
        n = dim
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        # Build simplex around best
        scale = ranges * 0.002
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_x.copy()
        f_simplex = np.zeros(n + 1)
        f_simplex[0] = best
        for i in range(n):
            simplex[i + 1] = best_x.copy()
            simplex[i + 1][i] += scale[i] if best_x[i] + scale[i] <= upper[i] else -scale[i]
            f_simplex[i + 1] = eval_f(simplex[i + 1])
        
        while time_left() > 0.05:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = eval_f(xr)
            
            if fr < f_simplex[0]:
                xe = clip(centroid + gamma * (xr - centroid))
                fe = eval_f(xe)
                if fe < fr:
                    simplex[-1], f_simplex[-1] = xe, fe
                else:
                    simplex[-1], f_simplex[-1] = xr, fr
            elif fr < f_simplex[-2]:
                simplex[-1], f_simplex[-1] = xr, fr
            else:
                xc = clip(centroid + rho * (simplex[-1] - centroid))
                fc = eval_f(xc)
                if fc < f_simplex[-1]:
                    simplex[-1], f_simplex[-1] = xc, fc
                else:
                    for
