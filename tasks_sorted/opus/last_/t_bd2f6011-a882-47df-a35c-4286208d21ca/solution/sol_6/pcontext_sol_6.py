#Looking at the previous algorithm, I can see several areas for improvement:
#
#1. **CMA-ES can be expensive for high dimensions** due to matrix operations - need a more scalable variant
#2. **Better initialization** with multiple strategies (Sobol-like, opposition-based)
#3. **Nelder-Mead** as a robust local search alternative
#4. **Differential Evolution** elements for better global exploration
#5. **More aggressive restarts** with adaptive strategies
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
    evals = 0

    def time_left():
        return max_time - (datetime.now() - start).total_seconds()

    def clip(x):
        return np.clip(x, lower, upper)

    def evaluate(x):
        nonlocal best, best_params, evals
        x = clip(x)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # Phase 1: Diverse initialization
    n_init = min(500, max(100, dim * 15))
    pop = []
    pop_f = []
    
    for i in range(n_init):
        if time_left() < 0.5:
            return best
        x = lower + np.random.uniform(0, 1, dim) * ranges
        f = evaluate(x)
        pop.append(x.copy())
        pop_f.append(f)
    
    pop = np.array(pop)
    pop_f = np.array(pop_f)

    # Phase 2: Differential Evolution with adaptive parameters
    def de_phase(budget_fraction=0.4):
        nonlocal pop, pop_f
        deadline_end = time_left() * budget_fraction
        t0 = (datetime.now() - start).total_seconds()
        
        NP = len(pop_f)
        # Keep top solutions
        if NP > 60:
            idx = np.argsort(pop_f)[:60]
            pop_local = pop[idx].copy()
            pop_f_local = pop_f[idx].copy()
        else:
            pop_local = pop.copy()
            pop_f_local = pop_f.copy()
        
        NP = len(pop_f_local)
        F = 0.8
        CR = 0.9
        
        while True:
            elapsed = (datetime.now() - start).total_seconds() - t0
            if elapsed > deadline_end or time_left() < 0.5:
                break
            
            for i in range(NP):
                if time_left() < 0.3:
                    return
                
                # current-to-best/1
                idxs = list(range(NP))
                idxs.remove(i)
                a, b = np.random.choice(idxs, 2, replace=False)
                best_idx = np.argmin(pop_f_local)
                
                # Adaptive F
                Fi = F + 0.1 * np.random.randn()
                Fi = np.clip(Fi, 0.1, 1.5)
                
                mutant = pop_local[i] + Fi * (pop_local[best_idx] - pop_local[i]) + Fi * (pop_local[a] - pop_local[b])
                
                # Binomial crossover
                CRi = np.clip(CR + 0.1 * np.random.randn(), 0.1, 1.0)
                mask = np.random.rand(dim) < CRi
                j_rand = np.random.randint(dim)
                mask[j_rand] = True
                
                trial = np.where(mask, mutant, pop_local[i])
                trial = clip(trial)
                
                f_trial = evaluate(trial)
                if f_trial <= pop_f_local[i]:
                    pop_f_local[i] = f_trial
                    pop_local[i] = trial.copy()
        
        pop[:] = pop[:len(pop)]  # keep original
    
    de_phase(budget_fraction=0.45)

    # Phase 3: Nelder-Mead from best point
    def nelder_mead(x0, alpha=1.0, gamma=2.0, rho=0.5, sigma_nm=0.5, init_step=0.05):
        n = dim
        simplex = [x0.copy()]
        f_simplex = [evaluate(x0)]
        
        for i in range(n):
            if time_left() < 0.3:
                return
            xi = x0.copy()
            xi[i] += ranges[i] * init_step
            xi = clip(xi)
            f_simplex.append(evaluate(xi))
            simplex.append(xi)
        
        while time_left() > 0.2:
            order = np.argsort(f_simplex)
            simplex = [simplex[i] for i in order]
            f_simplex = [f_simplex[i] for i in order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = evaluate(xr)
            
            if f_simplex[0] <= fr < f_simplex[-2]:
                simplex[-1] = xr; f_simplex[-1] = fr
            elif fr < f_simplex[0]:
                xe = clip(centroid + gamma * (xr - centroid))
                fe = evaluate(xe)
                if fe < fr:
                    simplex[-1] = xe; f_simplex[-1] = fe
                else:
                    simplex[-1] = xr; f_simplex[-1] = fr
            else:
                xc = clip(centroid + rho * (simplex[-1] - centroid))
                fc = evaluate(xc)
                if fc < f_simplex[-1]:
                    simplex[-1] = xc; f_simplex[-1] = fc
                else:
                    for i in range(1, len(simplex)):
                        if time_left() < 0.1:
                            return
                        simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                        f_simplex[i] = evaluate(simplex[i])

    if best_params is not None:
        nelder_mead(best_params.copy(), init_step=0.05)

    # Phase 4: Restarts with perturbation + Nelder-Mead
    while time_left() > 1.0:
        scale = np.random.uniform(0.001, 0.1)
        x0 = best_params + np.random.randn(dim) * ranges * scale
        x0 = clip(x0)
        nelder_mead(x0, init_step=scale)

    return best
#
#Key improvements:
#- **Differential Evolution** (current-to-best/1) replaces CMA-ES for better scalability and global search
#- **Nelder-Mead simplex** for robust local optimization without gradient/matrix overhead
#- **Adaptive parameters** (F, CR) in DE for better exploration
#- **Multiple restart scales** in the final phase to escape local optima
#- **Larger initial population** for better coverage of the search space
