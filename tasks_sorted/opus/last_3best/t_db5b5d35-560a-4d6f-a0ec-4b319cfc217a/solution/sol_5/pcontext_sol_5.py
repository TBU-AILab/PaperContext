#Looking at the results, algorithm 1 (output 10.28) performed best. I'll improve it by:
#
#1. **Better time allocation** - more time for DE, smarter local search scheduling
#2. **Improved mutation strategies** - adding current-to-rand/1 for diversity
#3. **Better Nelder-Mead with adaptive restarts** from top population members
#4. **Golden section search** as final polishing instead of simple coordinate descent
#5. **Opposition-based learning** for initialization diversity
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
    
    def evaluate(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # ---- Phase 1: LHS + Opposition-based initialization ----
    pop_size_init = min(max(10 * dim, 40), 180)
    pop_size = pop_size_init
    min_pop_size = max(4, dim // 2)
    
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.90:
            return best
        fitness[i] = evaluate(population[i])
    
    # Opposition-based: evaluate opposites of worst half, replace if better
    order = np.argsort(fitness)
    half = pop_size // 2
    for i in range(half, pop_size):
        if elapsed() >= max_time * 0.85:
            break
        idx = order[i]
        opp = lower + upper - population[idx]
        opp = np.clip(opp, lower, upper)
        f_opp = evaluate(opp)
        if f_opp < fitness[idx]:
            population[idx] = opp
            fitness[idx] = f_opp

    # ---- Phase 2: L-SHADE ----
    memory_size = 6
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    k = 0
    archive = []
    max_archive = pop_size_init
    
    stagnation = 0
    prev_best = best
    de_time_frac = 0.62
    
    while elapsed() < max_time * de_time_frac:
        S_F, S_CR, S_delta = [], [], []
        new_pop = population.copy()
        new_fit = fitness.copy()
        sorted_idx = np.argsort(fitness)
        
        for i in range(pop_size):
            if elapsed() >= max_time * de_time_frac:
                break
            
            ri = np.random.randint(memory_size)
            Fi = -1
            while Fi <= 0:
                Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
            Fi = min(Fi, 1.0)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            p_count = max(2, int(np.ceil(np.random.uniform(2.0/pop_size, 0.2) * pop_size)))
            pbest_idx = np.random.choice(sorted_idx[:p_count])
            
            candidates = list(range(pop_size)); candidates.remove(i)
            r1 = np.random.choice(candidates)
            combined_size = pop_size + len(archive)
            r2 = np.random.randint(combined_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(combined_size)
            xr2 = archive[r2 - pop_size] if r2 >= pop_size else population[r2]
            
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
            mutant = np.where(mutant < lower, (lower + population[i]) / 2, np.where(mutant > upper, (upper + population[i]) / 2, mutant))
            
            mask = (np.random.random(dim) < CRi) | (np.arange(dim) == np.random.randint(dim))
            trial = np.where(mask, mutant, population[i])
            
            f_trial = evaluate(trial)
            if f_trial < fitness[i]:
                S_F.append(Fi); S_CR.append(CRi); S_delta.append(fitness[i] - f_trial)
                if len(archive) < max_archive: archive.append(population[i].copy())
                elif archive: archive[np.random.randint(len(archive))] = population[i].copy()
                new_pop[i] = trial; new_fit[i] = f_trial
            elif f_trial == fitness[i]:
                new_pop[i] = trial; new_fit[i] = f_trial
        
        population = new_pop; fitness = new_fit
        if S_F:
            w = np.array(S_delta); w /= (w.sum() + 1e-30)
            sf = np.array(S_F); sc = np.array(S_CR)
            M_F[k] = np.sum(w * sf * sf) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * sc); k = (k + 1) % memory_size
        
        new_ps = max(min_pop_size, int(round(pop_size_init - (pop_size_init - min_pop_size) * elapsed() / (max_time * de_time_frac))))
        if new_ps < pop_size:
            si = np.argsort(fitness); population = population[si[:new_ps]]; fitness = fitness[si[:new_ps]]; pop_size = new_ps
        
        stagnation = stagnation + 1 if abs(prev_best - best) < 1e-14 else 0; prev_best = best
        if stagnation > 20:
            keep = max(pop_size // 4, 2); order = np.argsort(fitness)
            for ii in range(keep, pop_size):
                if elapsed() >= max_time * de_time_frac: break
                population[order[ii]] = best_params + np.random.randn(dim) * 0.1 * ranges if np.random.random() < 0.5 else lower + np.random.random(dim) * ranges
                population[order[ii]] = np.clip(population[order[ii]], lower, upper); fitness[order[ii]] = evaluate(population[order[ii]])
            stagnation = 0

    # ---- Phase 3: Nelder-Mead from best + top individuals ----
    def nelder_mead(x0, sf, tl):
        n=dim; scale=sf*ranges; simplex=np.zeros((n+1,n)); simplex[0]=x0.copy()
        for i in range(n): simplex[i+1]=x0.copy(); simplex[i+1][i]+=scale[i]*(1 if np.random.random()>0.5 else -1)
        simplex=np.clip(simplex,lower,upper); fs=np.array([evaluate(simplex[i]) for i in range(n+1) if elapsed()<tl])
        if len(fs)<n+1: return
        for _ in range(10000):
            if elapsed()>=tl: break
            o=np.argsort(fs);simplex=simplex[o];fs=fs[o];c=np.mean(simplex[:-1],axis=0)
            xr=np.clip(2*c-simplex[-1],lower,upper);fr=evaluate(xr)
            if fs[0]<=fr<fs[-2]:simplex[-1]=xr;fs[-1]=fr;continue
            if fr<fs[0]:
                xe=np.clip(c+2*(xr-c),lower,upper);fe=evaluate(xe)
                if fe<fr:simplex[-1]=xe;fs[-1]=fe
                else:simplex[-1]=xr;fs[-1]=fr
                continue
            if fr<fs[-1]:xc=np.clip(c+0.5*(xr-c),lower,upper);fc=evaluate(xc)
            else:xc=np.clip(c+0.5*(simplex[-1]-c),lower,upper);fc=evaluate(xc)
            if fc<max(fr,fs[-1]):simplex[-1]=xc;fs[-1]=fc;continue
            for i in range(1,n+1):
                if elapsed()>=tl:break
                simplex[i]=np.clip(simplex[0]+0.5*(simplex[i]-simplex[0]),lower,upper);fs[i]=evaluate(simplex[i])
            if np.max(np.abs(fs-fs[0]))<1e-16:break

    if best_params is not None:
        # Multi-start NM from top solutions
        top_idx = np.argsort(fitness)[:min(3, pop_size)]
        starts = [best_params.copy()] + [population[ti].copy() for ti in top_idx]
        for sf in [0.08, 0.015, 0.003, 0.0005]:
            for sx in starts:
                if elapsed()>=max_time*0.92: break
                nelder_mead(sx, sf, min(elapsed()+(max_time*0.92-elapsed())/(4*len(starts)), max_time*0.92))
            if elapsed()>=max_time*0.92: break

    # ---- Phase 4: Golden section coordinate descent ----
    if best_params is not None:
        x=best_params.copy();fx=best;gr=(np.sqrt(5)+1)/2
        for ss in [0.01,0.002,0.0003]:
            if elapsed()>=max_time*0.99:break
            for d in range(dim):
                if elapsed()>=max_time*0.99:break
                a=max(lower[d],x[d]-ss*ranges[d]);b=min(upper[d],x[d]+ss*ranges[d])
                c=b-(b-a)/gr;d2=a+(b-a)/gr
                xc=x.copy();xc[d]=c;fc=evaluate(xc)
                xd=x.copy();xd[d]=d2;fd=evaluate(xd)
                for _ in range(20):
                    if elapsed()>=max_time*0.99 or b-a<1e-14:break
                    if fc<fd:b=d2;d2=c;fd=fc;c=b-(b-a)/gr;xc=x.copy();xc[d]=c;fc=evaluate(xc)
                    else:a=c;c=d2;fc=fd;d2=a+(b-a)/gr;xd=x.copy();xd[d]=d2;fd=evaluate(xd)
                x[d]=(a+b)/2;xt=x.copy();ft=evaluate(xt)
                if ft<fx:fx=ft
    return best
