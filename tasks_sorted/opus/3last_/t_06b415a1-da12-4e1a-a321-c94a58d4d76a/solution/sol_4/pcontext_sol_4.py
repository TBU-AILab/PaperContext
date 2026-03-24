#Looking at the results, algorithm 1 (score 20.53) performs best. I'll analyze its strengths and weaknesses to improve further:
#
#**Key improvements:**
#1. Better time allocation - more time for CMA-ES which seems most effective
#2. Larger CMA-ES population for better exploration
#3. Multiple independent CMA-ES restarts with diverse starting points (not just from DE top)
#4. Improved L-SHADE with better parameter adaptation
#5. Nelder-Mead simplex as additional local search
#6. Smarter stagnation detection in CMA-ES
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
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_ok(fraction=0.97):
        return elapsed() < max_time * fraction
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    eval_count = [0]
    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        eval_count[0] += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # --- Phase 1: LHS Initialization ---
    pop_size_init = min(max(16 * dim, 80), 350)
    pop_size = pop_size_init
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            pop[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if not time_ok(0.95):
            return best
        fitness[i] = evaluate(pop[i])
    
    # Opposition-based population
    opp_pop = lower + upper - pop
    opp_fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if not time_ok(0.93):
            break
        opp_fitness[i] = evaluate(opp_pop[i])
    
    all_pop = np.vstack([pop, opp_pop])
    all_fit = np.concatenate([fitness, opp_fitness])
    order = np.argsort(all_fit)[:pop_size]
    pop = all_pop[order].copy()
    fitness = all_fit[order].copy()
    
    # --- Phase 2: L-SHADE ---
    memory_size = 6
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    k = 0
    archive = []
    nfe = 0
    max_nfe_shade = pop_size_init * 180
    
    while time_ok(0.35):
        S_F, S_CR, S_df = [], [], []
        sorted_idx = np.argsort(fitness)
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if not time_ok(0.35):
                break
            
            ri = np.random.randint(memory_size)
            F_i = -1
            while F_i <= 0:
                F_i = np.random.standard_cauchy() * 0.1 + M_F[ri]
                if F_i >= 1.0:
                    F_i = 1.0
                    break
            F_i = min(F_i, 1.0)
            
            if M_CR[ri] < 0:
                CR_i = 0.0
            else:
                CR_i = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            p = max(2, int(np.random.uniform(0.05, 0.2) * pop_size))
            pbest_idx = sorted_idx[np.random.randint(p)]
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            pool = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            pool = [x for x in pool if x != i and x != r1]
            if not pool:
                pool = [xx for xx in range(pop_size) if xx != i]
            r2_idx = np.random.choice(pool)
            xr2 = pop[r2_idx] if r2_idx < pop_size else archive[r2_idx - pop_size]
            
            mutant = pop[i] + F_i * (pop[pbest_idx] - pop[i]) + F_i * (pop[r1] - xr2)
            
            trial = pop[i].copy()
            j_rand = np.random.randint(dim)
            mask = (np.random.random(dim) < CR_i)
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            for j in range(dim):
                if trial[j] < lower[j]:
                    trial[j] = (lower[j] + pop[i][j]) / 2
                elif trial[j] > upper[j]:
                    trial[j] = (upper[j] + pop[i][j]) / 2
            
            f_trial = evaluate(trial)
            nfe += 1
            
            if f_trial <= fitness[i]:
                if f_trial < fitness[i]:
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    S_df.append(abs(fitness[i] - f_trial))
                    archive.append(pop[i].copy())
                new_pop[i] = clip(trial)
                new_fitness[i] = f_trial
        
        pop = new_pop
        fitness = new_fitness
        
        if S_F:
            w = np.array(S_df)
            w = w / (w.sum() + 1e-30)
            sf = np.array(S_F)
            scr = np.array(S_CR)
            M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * scr)
            k = (k + 1) % memory_size
        
        while len(archive) > pop_size:
            archive.pop(np.random.randint(len(archive)))
        
        new_pop_size = max(4, int(round(pop_size_init - (pop_size_init - 4) * nfe / max_nfe_shade)))
        if new_pop_size < pop_size:
            order = np.argsort(fitness)[:new_pop_size]
            pop = pop[order]
            fitness = fitness[order]
            pop_size = new_pop_size
    
    # --- Phase 3: Multiple CMA-ES restarts ---
    top_k = min(10, len(pop))
    top_indices = np.argsort(fitness)[:top_k]
    
    restart = 0
    while time_ok(0.88) and restart < top_k + 5:
        if restart < top_k:
            x0 = pop[top_indices[restart]].copy()
        else:
            x0 = best_params + np.random.randn(dim) * ranges * 0.05
            x0 = clip(x0)
        restart += 1
        
        n = dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        cc = (4+mueff/n)/(n+4+2*mueff/n)
        cs = (mueff+2)/(n+mueff+5)
        c1 = 2/((n+1.3)**2+mueff)
        cmu_v = min(1-c1, 2*(mueff-2+1/mueff)/((n+2)**2+mueff))
        damps = 1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs
        chiN = np.sqrt(n)*(1-1/(4*n)+1/(21*n**2))
        mean = x0.copy()
        sigma = np.mean(ranges)*0.15/(1+restart*0.2)
        pc=np.zeros(n);ps=np.zeros(n);C=np.eye(n);counteval=0;stag=0;best_gen_f=float('inf')
        
        for gen in range(3000):
            if not time_ok(0.88): break
            try:
                C=(C+C.T)/2; D_vals,B=np.linalg.eigh(C); D=np.sqrt(np.maximum(D_vals,1e-20))
            except: break
            arx=np.zeros((lam,n));arf=np.zeros(lam)
            for kk in range(lam):
                if not time_ok(0.88): break
                arx[kk]=clip(mean+sigma*(B@(D*np.random.randn(n))))
                arf[kk]=evaluate(arx[kk]);counteval+=1
            idx=np.argsort(arf);old_mean=mean.copy()
            mean=np.sum(weights[:,None]*arx[idx[:mu]],axis=0)
            ymean=(mean-old_mean)/sigma
            ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*(B@(1.0/D*(B.T@ymean)))
            hsig=float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*counteval/lam))/chiN<1.4+2/(n+1))
            pc=(1-cc)*pc+hsig*np.sqrt(cc*(2-cc)*mueff)*ymean
            artmp=(arx[idx[:mu]]-old_mean)/sigma
            C=(1-c1-cmu_v)*C+c1*(np.outer(pc,pc)+(1-hsig)*cc*(2-cc)*C)+cmu_v*(artmp.T@np.diag(weights)@artmp)
            sigma*=np.exp((cs/damps)*(np.linalg.norm(ps)/chiN-1))
            sigma=np.clip(sigma,1e-16,np.max(ranges))
            cf=arf[idx[0]]
            if cf<best_gen_f-1e-10: best_gen_f=cf;stag=0
            else: stag+=1
            if stag>15+30*n//lam or sigma<1e-14: break
    
    # --- Phase 4: Coordinate descent ---
    if best_params is not None:
        x=best_params.copy();f_x=best;step=ranges*0.01
        while time_ok(0.99):
            improved=False
            for d in range(dim):
                if not time_ok(0.99): break
                for sign in [1,-1]:
                    trial=x.copy();trial[d]+=sign*step[d];trial=clip(trial)
                    ft=evaluate(trial)
                    if ft<f_x: x=trial;f_x=ft;improved=True;step[d]*=1.5;break
                else: step[d]*=0.5
            if not improved and np.max(step/ranges)<1e-15: break
    
    return best
