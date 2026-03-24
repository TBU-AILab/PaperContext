#Looking at the results, algorithm 2 performed best with DE + local search. I'll create an improved hybrid that combines:
#
#1. **L-SHADE** (superior adaptive DE) with opposition-based learning
#2. **CMA-ES** restarts for refinement
#3. **Pattern search** for final polish
#4. Better time management and parameter tuning
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

    # ---- L-SHADE Phase (50% of time) ----
    pop_size_init = min(max(60, 18 * dim), 600)
    pop_size = pop_size_init
    min_pop = 4
    
    # LHS initialization
    population = np.zeros((pop_size_init, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size_init)
        for i in range(pop_size_init):
            population[i, d] = lower[d] + (perm[i] + np.random.rand()) / pop_size_init * ranges[d]
    
    fitness = np.array([eval_f(ind) for ind in population])
    
    # Opposition-based population
    if time_left() > max_time * 0.8:
        opp = lower + upper - population
        opp_fit = np.array([eval_f(ind) for ind in opp])
        all_pop = np.vstack([population, opp])
        all_fit = np.concatenate([fitness, opp_fit])
        idx = np.argsort(all_fit)[:pop_size_init]
        population = all_pop[idx]
        fitness = all_fit[idx]
    
    memory_size = 10
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.8)
    mem_idx = 0
    archive = []
    archive_max = pop_size_init
    shade_end = max_time * 0.50
    
    while elapsed() < shade_end and time_left() > 0.5:
        S_F, S_CR, delta_f = [], [], []
        p_best_rate = max(2.0 / pop_size, 0.05 + 0.1 * (1 - elapsed() / shade_end))
        
        for i in range(pop_size):
            if elapsed() >= shade_end or time_left() < 0.3:
                break
            ri = np.random.randint(0, memory_size)
            Fi = -1
            while Fi <= 0:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 1: Fi = 1.0
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            p = max(2, int(p_best_rate * pop_size))
            sorted_idx = np.argsort(fitness)
            pbest = sorted_idx[np.random.randint(0, p)]
            
            cands = list(range(pop_size))
            cands.remove(i)
            r1 = cands[np.random.randint(len(cands))]
            
            pool2 = [j for j in range(pop_size) if j != i and j != r1]
            r2c = np.random.randint(len(pool2) + len(archive))
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
            
            if tf <= fitness[i]:
                if tf < fitness[i]:
                    S_F.append(Fi)
                    S_CR.append(CRi)
                    delta_f.append(fitness[i] - tf)
                    archive.append(population[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(np.random.randint(len(archive)))
                population[i] = trial
                fitness[i] = tf
        
        if S_F:
            w = np.array(delta_f)
            w = w / (w.sum() + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[mem_idx] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[mem_idx] = np.sum(w * scr)
            mem_idx = (mem_idx + 1) % memory_size
        
        progress = elapsed() / shade_end
        new_ps = max(min_pop, int(pop_size_init - (pop_size_init - min_pop) * progress))
        if new_ps < pop_size:
            sidx = np.argsort(fitness)
            population = population[sidx[:new_ps]]
            fitness = fitness[sidx[:new_ps]]
            pop_size = new_ps

    # ---- CMA-ES Phase (40% of time) ----
    def cma_es(x0, sigma0, max_t):
        nonlocal best, best_x
        n = dim
        lam = 4 + int(3 * np.log(n))
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
        xmean = x0.copy(); sigma = sigma0
        pc = np.zeros(n); ps = np.zeros(n)
        C = np.eye(n); B = np.eye(n); D = np.ones(n)
        invsqrtC = np.eye(n); evals = 0; eigeneval = 0
        t_end = elapsed() + max_t
        while elapsed() < t_end and time_left() > 0.05:
            arx = np.array([clip(xmean + sigma*(B@(D*np.random.randn(n)))) for _ in range(lam)])
            fit = np.array([eval_f(arx[k]) for k in range(lam)]); evals += lam
            idx = np.argsort(fit); xold = xmean.copy()
            xmean = arx[idx[:mu]].T @ weights
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff)*(invsqrtC@((xmean-xold)/sigma))
            hsig = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*evals/lam))/chiN < 1.4+2/(n+1)
            pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mueff)*((xmean-xold)/sigma)
            artmp = (arx[idx[:mu]]-xold)/sigma
            C = (1-c1-cmu_v)*C + c1*(np.outer(pc,pc)+(1-hsig)*cc*(2-cc)*C) + cmu_v*(artmp.T@np.diag(weights)@artmp)
            sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN-1))
            if evals-eigeneval > lam/(c1+cmu_v)/n/10:
                eigeneval=evals; C=np.triu(C)+np.triu(C,1).T
                try: D2,B=np.linalg.eigh(C); D=np.sqrt(np.maximum(D2,1e-20)); invsqrtC=B@np.diag(1/D)@B.T
                except: break
            if sigma*np.max(D)<1e-14*np.max(ranges): break

    for sigma_f in [0.2, 0.02, 0.002]:
        if time_left() < 0.5: break
        cma_es(best_x.copy(), np.mean(ranges)*sigma_f, min(time_left()*0.4, time_left()-0.3))

    # ---- Pattern search polish (remaining time) ----
    if best_x is not None:
        cur = best_x.copy(); cur_f = best
        step = ranges * 0.005
        while time_left() > 0.05:
            improved = False
            for j in range(dim):
                if time_left() < 0.05: break
                for sign in [1, -1]:
                    t = cur.copy(); t[j] = np.clip(cur[j]+sign*step[j], lower[j], upper[j])
                    tf = eval_f(t)
                    if tf < cur_f: cur=t; cur_f=tf; improved=True; break
            if not improved:
                step *= 0.5
                if np.max(step/ranges) < 1e-12: break
    
    return best
