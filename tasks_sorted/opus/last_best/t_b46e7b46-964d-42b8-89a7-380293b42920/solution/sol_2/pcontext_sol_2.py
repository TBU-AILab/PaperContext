#Looking at the previous algorithm, I'll make significant improvements:
#
#1. **L-SHADE with linear population size reduction** - proven more effective
#2. **Multiple restarts** to escape local optima
#3. **Better local search** with coordinate descent + Nelder-Mead
#4. **Opposition-based learning** for initialization
#5. **Improved parameter adaptation**
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    mid = (lower + upper) / 2
    
    evals = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
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

    def lshade_run(time_fraction_end, pop_init=None):
        nonlocal best, best_params
        
        N_init = min(max(25, 6 * dim), 200)
        N_min = 4
        
        if pop_init is not None:
            population = pop_init.copy()
            N_init = len(population)
        else:
            # LHS + opposition
            population = np.zeros((N_init, dim))
            for d in range(dim):
                perm = np.random.permutation(N_init)
                for i in range(N_init):
                    population[i, d] = lower[d] + (perm[i] + np.random.random()) / N_init * ranges[d]
        
        pop_size = len(population)
        fitness = np.array([evaluate(population[i]) for i in range(pop_size) if elapsed() < max_time * time_fraction_end])
        if len(fitness) < pop_size:
            fitness = np.append(fitness, [float('inf')] * (pop_size - len(fitness)))
            
        # Opposition-based: evaluate opposites, keep best N_init
        if pop_init is None and elapsed() < max_time * time_fraction_end:
            opp = lower + upper - population
            opp_fitness = np.array([evaluate(opp[i]) for i in range(pop_size) if elapsed() < max_time * time_fraction_end])
            if len(opp_fitness) == pop_size:
                combined = np.vstack([population, opp])
                combined_f = np.concatenate([fitness, opp_fitness])
                idx = np.argsort(combined_f)[:pop_size]
                population = combined[idx]
                fitness = combined_f[idx]
        
        mem_size = 6
        M_F = np.full(mem_size, 0.5)
        M_CR = np.full(mem_size, 0.8)
        k = 0
        archive = []
        gen = 0
        max_gen_est = max(1, int((max_time * time_fraction_end - elapsed()) / (pop_size * 0.001 + 0.01)))
        
        while elapsed() < max_time * time_fraction_end and pop_size >= N_min:
            S_F, S_CR, S_w = [], [], []
            
            # Linear pop reduction
            gen += 1
            ratio = min(1.0, elapsed() / (max_time * time_fraction_end))
            new_pop_size = max(N_min, int(round(N_init - (N_init - N_min) * ratio)))
            
            for i in range(pop_size):
                if elapsed() >= max_time * time_fraction_end:
                    break
                
                ri = np.random.randint(mem_size)
                Fi = -1
                while Fi <= 0:
                    Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                Fi = min(Fi, 1.0)
                
                CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
                
                p = max(2, int(max(0.05, 0.2 - 0.15 * ratio) * pop_size))
                pbest_idx = np.argsort(fitness)[:p]
                xpbest = population[np.random.choice(pbest_idx)]
                
                idxs = [j for j in range(pop_size) if j != i]
                a = np.random.choice(idxs)
                
                pool = list(range(pop_size))
                pool.remove(i)
                if a in pool: pool.remove(a)
                if archive:
                    b_src = np.vstack([population[pool], np.array(archive)])
                    b_idx = np.random.randint(len(b_src))
                    xb = b_src[b_idx]
                else:
                    xb = population[np.random.choice(pool)] if pool else population[a]
                
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
                        S_F.append(Fi); S_CR.append(CRi)
                        S_w.append(abs(fitness[i] - tf))
                        archive.append(population[i].copy())
                        if len(archive) > N_init:
                            archive.pop(np.random.randint(len(archive)))
                    population[i] = trial
                    fitness[i] = tf
            
            if S_F:
                w = np.array(S_w); w /= w.sum() + 1e-30
                sf = np.array(S_F)
                M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
                M_CR[k] = np.sum(w * np.array(S_CR))
                k = (k + 1) % mem_size
            
            if new_pop_size < pop_size:
                idx = np.argsort(fitness)[:new_pop_size]
                population = population[idx]
                fitness = fitness[idx]
                pop_size = new_pop_size

    # Run L-SHADE
    lshade_run(0.70)
    
    # Restart L-SHADE from perturbed best
    if best_params is not None and elapsed() < max_time * 0.85:
        n_restart = min(max(20, 4*dim), 100)
        pop2 = np.array([best_params + np.random.normal(0, 0.1, dim) * ranges for _ in range(n_restart)])
        pop2 = clip(pop2)
        lshade_run(0.88, pop_init=pop2)
    
    # Coordinate descent local search
    if best_params is not None and elapsed() < max_time * 0.95:
        x = best_params.copy()
        fx = best
        step = 0.01 * ranges
        for _ in range(20):
            if elapsed() >= max_time * 0.95: break
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.95: break
                for sign in [1, -1]:
                    xn = x.copy(); xn[d] = np.clip(x[d] + sign*step[d], lower[d], upper[d])
                    fn = evaluate(xn)
                    if fn < fx:
                        x, fx = xn, fn; improved = True
                        while elapsed() < max_time * 0.95:
                            xn = x.copy(); xn[d] = np.clip(x[d]+sign*step[d]*2, lower[d], upper[d])
                            fn = evaluate(xn)
                            if fn < fx: x, fx = xn, fn
                            else: break
                        break
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-10: break

    # Final Nelder-Mead
    if best_params is not None and elapsed() < max_time * 0.99:
        n = dim; scale = 0.005 * ranges
        simplex = np.zeros((n+1, n)); fs = np.zeros(n+1)
        simplex[0] = best_params.copy(); fs[0] = best
        for i in range(1, n+1):
            if elapsed() >= max_time*0.99: break
            simplex[i] = best_params.copy()
            simplex[i][(i-1)%n] += scale[(i-1)%n]*(1 if np.random.random()>0.5 else -1)
            simplex[i] = clip(simplex[i]); fs[i] = evaluate(simplex[i])
        for _ in range(5000):
            if elapsed() >= max_time*0.99: break
            o = np.argsort(fs); simplex = simplex[o]; fs = fs[o]
            c = np.mean(simplex[:-1], axis=0)
            xr = clip(c+(c-simplex[-1])); fr = evaluate(xr)
            if fr < fs[0]:
                xe = clip(c+2*(xr-c)); fe = evaluate(xe)
                simplex[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < fs[-2]:
                simplex[-1] = xr; fs[-1] = fr
            else:
                xc = clip(c+0.5*(simplex[-1]-c)); fc = evaluate(xc)
                if fc < fs[-1]: simplex[-1] = xc; fs[-1] = fc
                else:
                    for i in range(1, n+1):
                        if elapsed()>=max_time*0.99: break
                        simplex[i] = simplex[0]+0.5*(simplex[i]-simplex[0])
                        simplex[i] = clip(simplex[i]); fs[i] = evaluate(simplex[i])
            if np.std(fs) < 1e-16: break
    
    return best
