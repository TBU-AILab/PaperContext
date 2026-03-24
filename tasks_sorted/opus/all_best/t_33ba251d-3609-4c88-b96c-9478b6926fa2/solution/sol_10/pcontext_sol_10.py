#Looking at the best algorithm (score 6.79), I'll analyze what worked and what can be improved:
#
#1. L-SHADE + multi-start Nelder-Mead is the winning formula
#2. Key improvements: better top-k diversity tracking, add coordinate descent between NM restarts, tune M_CR initialization higher, add inside contraction to NM, better time allocation, and adaptive perturbation in stagnation recovery
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
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

    # LHS initialization
    init_pop_size = min(max(18, 7 * dim), 140)
    population = np.zeros((init_pop_size, dim))
    for j in range(dim):
        perm = np.random.permutation(init_pop_size)
        for i in range(init_pop_size):
            population[i, j] = lower[j] + (perm[i] + np.random.rand()) / init_pop_size * ranges[j]
    
    fitness = np.zeros(init_pop_size)
    for i in range(init_pop_size):
        if elapsed() >= max_time * 0.90:
            return best
        fitness[i] = eval_f(population[i])
    
    pop_size = init_pop_size
    
    # Opposition-based learning on worst quarter
    idx_sort = np.argsort(fitness)
    n_opp = min(pop_size // 4, 25)
    for i in range(n_opp):
        if elapsed() >= max_time * 0.85:
            break
        wi = idx_sort[-(i+1)]
        opp = lower + upper - population[wi]
        f_opp = eval_f(opp)
        if f_opp < fitness[wi]:
            population[wi] = opp
            fitness[wi] = f_opp

    # L-SHADE parameters
    H = 80
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.85)
    mem_idx = 0
    archive = []
    archive_max = init_pop_size
    min_pop_size = max(4, dim // 3)
    
    stagnation = 0
    prev_best = best
    gen = 0
    
    # Track top-k diverse solutions for local search seeding
    top_k = []
    min_dist_thresh = 0.01 * np.linalg.norm(ranges)
    
    def update_topk(x, f, k=10):
        # Check if close to existing
        for idx_t in range(len(top_k)):
            if np.linalg.norm(x - top_k[idx_t][1]) < min_dist_thresh:
                if f < top_k[idx_t][0]:
                    top_k[idx_t] = (f, x.copy())
                    top_k.sort(key=lambda t: t[0])
                return
        top_k.append((f, x.copy()))
        top_k.sort(key=lambda t: t[0])
        while len(top_k) > k:
            top_k.pop()

    update_topk(best_x, best)

    # Main L-SHADE loop
    de_time_frac = 0.65
    while elapsed() < max_time * de_time_frac and pop_size >= min_pop_size:
        gen += 1
        S_F, S_CR, S_df = [], [], []
        sorted_idx = np.argsort(fitness)
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        time_ratio = elapsed() / (max_time * de_time_frac + 1e-30)
        p_rate = max(0.05, 0.25 - 0.20 * time_ratio)
        
        for i in range(pop_size):
            if elapsed() >= max_time * de_time_frac:
                break
            
            ri = np.random.randint(H)
            Fi = -1
            attempts = 0
            while Fi <= 0 and attempts < 15:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                attempts += 1
            if Fi <= 0: Fi = 0.5
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            p = max(2, int(p_rate * pop_size))
            xpbest = population[sorted_idx[np.random.randint(p)]]
            
            r1 = np.random.randint(pop_size - 1)
            if r1 >= i: r1 += 1
            
            union_size = pop_size + len(archive)
            r2 = np.random.randint(union_size - 1)
            if r2 >= i: r2 += 1
            if r2 == r1: r2 = (r2 + 1) % union_size
            xr2 = population[r2] if r2 < pop_size else archive[r2 - pop_size]
            
            mutant = population[i] + Fi * (xpbest - population[i]) + Fi * (population[r1] - xr2)
            for d in range(dim):
                if mutant[d] < lower[d]:
                    mutant[d] = lower[d] + np.random.rand() * (population[i][d] - lower[d])
                elif mutant[d] > upper[d]:
                    mutant[d] = upper[d] - np.random.rand() * (upper[d] - population[i][d])
            
            mask = np.random.rand(dim) < CRi
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, population[i])
            
            f_trial = eval_f(trial)
            if f_trial <= fitness[i]:
                if f_trial < fitness[i]:
                    S_F.append(Fi); S_CR.append(CRi); S_df.append(fitness[i] - f_trial)
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(len(archive)))
                    update_topk(trial, f_trial)
                new_pop[i] = trial; new_fit[i] = f_trial
        
        population = new_pop; fitness = new_fit
        
        if S_F:
            w = np.array(S_df); w /= w.sum() + 1e-30
            sf_arr = np.array(S_F)
            M_F[mem_idx] = np.sum(w * sf_arr**2) / (np.sum(w * sf_arr) + 1e-30)
            M_CR[mem_idx] = np.sum(w * np.array(S_CR))
            mem_idx = (mem_idx + 1) % H
        
        new_size = max(min_pop_size, int(round(init_pop_size - (init_pop_size - min_pop_size) * elapsed() / (max_time * de_time_frac))))
        if new_size < pop_size:
            si = np.argsort(fitness)
            population = population[si[:new_size]]; fitness = fitness[si[:new_size]]; pop_size = new_size
        
        stagnation = stagnation + 1 if abs(best - prev_best) < 1e-14 else 0
        prev_best = best
        if stagnation > 10 + dim:
            half = pop_size // 2; si = np.argsort(fitness)
            for i in range(half, pop_size):
                if elapsed() >= max_time * de_time_frac: break
                scale = 0.01 + 0.09 * np.random.rand()
                population[si[i]] = clip(best_x + scale * ranges * np.random.randn(dim))
                fitness[si[i]] = eval_f(population[si[i]])
            stagnation = 0

    # Multi-start Nelder-Mead + coordinate descent
    nm_restart = 0
    while elapsed() < max_time * 0.97 and best_x is not None:
        seed_idx = nm_restart % max(1, len(top_k))
        seed_x = top_k[seed_idx][1] if nm_restart < len(top_k) * 3 else best_x
        n = min(dim + 1, 55)
        simplex = np.zeros((n, dim)); sf = np.zeros(n)
        simplex[0] = seed_x.copy(); sf[0] = eval_f(seed_x)
        sc = ranges * max(1e-6, 0.025 * (0.5 ** (nm_restart // 3)))
        for i in range(1, n): simplex[i] = clip(seed_x + sc * np.random.randn(dim)); sf[i] = eval_f(simplex[i])
        no_improve = 0
        while elapsed() < max_time * 0.97:
            idx = np.argsort(sf); simplex = simplex[idx]; sf = sf[idx]
            old_b = sf[0]; c = np.mean(simplex[:-1], axis=0)
            xr = clip(2*c - simplex[-1]); fr = eval_f(xr)
            if sf[0] <= fr < sf[-2]: simplex[-1]=xr; sf[-1]=fr
            elif fr < sf[0]:
                xe = clip(c+2*(xr-c)); fe = eval_f(xe)
                if fe<fr: simplex[-1]=xe; sf[-1]=fe
                else: simplex[-1]=xr; sf[-1]=fr
            else:
                if fr < sf[-1]:
                    xoc = clip(c+0.5*(xr-c)); foc = eval_f(xoc)
                    if foc<=fr: simplex[-1]=xoc; sf[-1]=foc
                    else:
                        for i in range(1,n): simplex[i]=clip(simplex[0]+0.5*(simplex[i]-simplex[0])); sf[i]=eval_f(simplex[i])
                else:
                    xic = clip(c+0.5*(simplex[-1]-c)); fic = eval_f(xic)
                    if fic<sf[-1]: simplex[-1]=xic; sf[-1]=fic
                    else:
                        for i in range(1,n): simplex[i]=clip(simplex[0]+0.5*(simplex[i]-simplex[0])); sf[i]=eval_f(simplex[i])
            no_improve = no_improve+1 if abs(sf[0]-old_b)<1e-15 else 0
            if no_improve > 3*dim: break
        nm_restart += 1
    
    return best
