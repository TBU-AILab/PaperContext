#Looking at the code, I'll focus on several key improvements:
#
#1. **CMA-ES inspired local search** instead of just Nelder-Mead - better for rotated landscapes
#2. **More restarts with diverse strategies** to escape local optima
#3. **Better time management** - spend more time on exploitation
#4. **Improved L-SHADE with better parameter adaptation**
#5. **Pattern search as final polish**
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
    center = (lower + upper) / 2
    
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
    pop_size_init = min(max(18 * dim, 60), 250)
    pop_size_min = max(4, dim)
    pop_size = pop_size_init
    H = 80
    
    # Opposition-based LHS initialization
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
        if elapsed() >= max_time * 0.85:
            return best
        all_fit[i] = evaluate(all_pop[i])
    
    order = np.argsort(all_fit)[:pop_size]
    pop = all_pop[order].copy()
    fit = all_fit[order].copy()
    
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.8)
    k = 0
    archive = []
    
    while elapsed() < max_time * 0.55:
        S_F, S_CR, S_w = [], [], []
        progress = elapsed() / (max_time * 0.55)
        p_rate = max(0.05, 0.25 - 0.20 * progress)
        
        new_pop = pop.copy()
        new_fit = fit.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.55:
                break
            
            ri = np.random.randint(H)
            Fi = -1
            for _ in range(10):
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 0:
                    break
            Fi = np.clip(Fi, 0.05, 1.0)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            p = max(2, int(p_rate * pop_size))
            p = min(p, pop_size)
            top_p = np.argpartition(fit, min(p, pop_size-1))[:p]
            xp = pop[top_p[np.random.randint(len(top_p))]]
            
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1 = candidates[np.random.randint(len(candidates))]
            
            union = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            union = [x for x in union if x != i and x != r1]
            if not union:
                union = [r1]
            r2_idx = union[np.random.randint(len(union))]
            xr2 = pop[r2_idx] if r2_idx < pop_size else archive[r2_idx - pop_size]
            
            mutant = pop[i] + Fi * (xp - pop[i]) + Fi * (pop[r1] - xr2)
            
            mask = np.random.random(dim) < CRi
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, pop[i])
            
            below = trial < lower; above = trial > upper
            trial[below] = (lower[below] + pop[i][below]) / 2
            trial[above] = (upper[above] + pop[i][above]) / 2
            trial = np.clip(trial, lower, upper)
            
            f_trial = evaluate(trial)
            if f_trial <= fit[i]:
                if f_trial < fit[i]:
                    S_F.append(Fi); S_CR.append(CRi); S_w.append(abs(fit[i] - f_trial))
                    if len(archive) < pop_size:
                        archive.append(pop[i].copy())
                    elif archive:
                        archive[np.random.randint(len(archive))] = pop[i].copy()
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

    # --- CMA-ES inspired local search ---
    if best_params is not None and elapsed() < max_time * 0.95:
        sigma = 0.05
        lam = max(4 + int(3 * np.log(dim)), 8)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2*(mueff - 2 + 1/mueff)/((dim+2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(dim+1))-1) + cs
        
        mean = best_params.copy()
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        for gen in range(5000):
            if elapsed() >= max_time * 0.96:
                break
            try:
                sqrtC = np.linalg.cholesky(C)
            except:
                C = np.eye(dim)
                sqrtC = np.eye(dim)
            
            arz = np.random.randn(lam, dim)
            arx = mean + sigma * (arz @ sqrtC.T)
            arx = np.clip(arx, lower, upper)
            
            arfitness = np.array([evaluate(arx[j]) for j in range(lam) if elapsed() < max_time * 0.96])
            if len(arfitness) < lam:
                break
            
            idx = np.argsort(arfitness)
            mean_old = mean.copy()
            mean = np.sum(weights[:, None] * arx[idx[:mu]], axis=0)
            
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * np.linalg.solve(sqrtC, (mean - mean_old)/sigma)
            hsig = float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(gen+1)))/chiN < 1.4 + 2/(dim+1))
            pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mueff)*(mean - mean_old)/sigma
            
            artmp = (arx[idx[:mu]] - mean_old) / sigma
            C = (1-c1-cmu_val)*C + c1*(np.outer(pc,pc) + (1-hsig)*cc*(2-cc)*C) + cmu_val * (weights[:, None] * artmp).T @ artmp
            C = (C + C.T) / 2
            sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
            sigma = np.clip(sigma, 1e-20, np.max(ranges))

    # --- Final Nelder-Mead polish ---
    if best_params is not None:
        for sf_val in [0.01, 0.002, 0.0005]:
            if elapsed() >= max_time * 0.99: break
            n = dim; scale = sf_val * ranges
            simplex = np.zeros((n+1, n)); sfit = np.zeros(n+1)
            simplex[0] = best_params.copy(); sfit[0] = best
            for i in range(n):
                if elapsed() >= max_time * 0.99: return best
                simplex[i+1] = best_params.copy()
                simplex[i+1][i] += scale[i] if best_params[i]+scale[i]<=upper[i] else -scale[i]
                sfit[i+1] = evaluate(simplex[i+1])
            for _ in range(300*dim):
                if elapsed() >= max_time*0.99: break
                o=np.argsort(sfit); simplex=simplex[o]; sfit=sfit[o]
                c=np.mean(simplex[:-1],axis=0)
                xr=np.clip(c+(c-simplex[-1]),lower,upper); fr=evaluate(xr)
                if elapsed()>=max_time*0.99: break
                if sfit[0]<=fr<sfit[-2]: simplex[-1]=xr; sfit[-1]=fr
                elif fr<sfit[0]:
                    xe=np.clip(c+2*(xr-c),lower,upper); fe=evaluate(xe)
                    if fe<fr: simplex[-1]=xe; sfit[-1]=fe
                    else: simplex[-1]=xr; sfit[-1]=fr
                else:
                    xc=np.clip(c+0.5*(simplex[-1]-c),lower,upper); fc=evaluate(xc)
                    if fc<sfit[-1]: simplex[-1]=xc; sfit[-1]=fc
                    else:
                        for i in range(1,n+1):
                            if elapsed()>=max_time*0.99: break
                            simplex[i]=simplex[0]+0.5*(simplex[i]-simplex[0])
                            simplex[i]=np.clip(simplex[i],lower,upper); sfit[i]=evaluate(simplex[i])
    return best
