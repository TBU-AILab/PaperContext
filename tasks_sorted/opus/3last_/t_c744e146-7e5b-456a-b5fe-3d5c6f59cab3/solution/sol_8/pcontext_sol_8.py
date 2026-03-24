#Looking at the results, I need to improve the optimization algorithm. The best so far achieved 26.28. Let me analyze what can be improved:
#
#1. Better initial sampling and population diversity
#2. More robust L-SHADE with proper parameter adaptation
#3. Better CMA-ES implementation with proper restarts
#4. More sophisticated local search
#5. Better time budget allocation
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    evals = [0]
    
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
        evals[0] += 1
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

    # ---- CMA-ES with sep-CMA fallback for high dim ----
    def cma_es(x0, sigma0, max_t, lam_mult=1.0):
        nonlocal best, best_x
        n = dim
        use_sep = n > 40  # Use separable CMA for high dimensions
        
        lam = max(int((4 + int(3 * np.log(n))) * lam_mult), 6)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + cs
        
        if use_sep:
            c1 *= (n + 2) / 3.0
            cmu_val *= min(1, (n + 2) / 3.0)
            c1 = min(c1, 0.9 * (1 - cmu_val))
        
        xmean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        if use_sep:
            diagC = np.ones(n)
        else:
            B = np.eye(n)
            D = np.ones(n)
            C = np.eye(n)
            invsqrtC = np.eye(n)
        
        eigeneval = 0
        chiN = n**0.5 * (1 - 1/(4*n) + 1/(21*n**2))
        
        local_evals = 0
        t_end = elapsed() + max_t
        best_gen_f = float('inf')
        stag_count = 0
        
        while elapsed() < t_end and time_left() > 0.05:
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            
            if use_sep:
                sqrtC = np.sqrt(diagC)
                for k in range(lam):
                    arx[k] = xmean + sigma * sqrtC * arz[k]
                    arx[k] = clip(arx[k])
            else:
                for k in range(lam):
                    arx[k] = xmean + sigma * (B @ (D * arz[k]))
                    arx[k] = clip(arx[k])
            
            fit = np.array([eval_f(arx[k]) for k in range(lam)])
            local_evals += lam
            
            idx = np.argsort(fit)
            arx = arx[idx]
            arz = arz[idx]
            
            gen_best = fit[idx[0]]
            if gen_best < best_gen_f - 1e-12:
                best_gen_f = gen_best
                stag_count = 0
            else:
                stag_count += 1
            
            xold = xmean.copy()
            xmean = arx[:mu].T @ weights
            
            zmean = arz[:mu].T @ weights
            
            if use_sep:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * zmean
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*local_evals/lam)) / chiN) < (1.4 + 2/(n+1))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ((xmean - xold) / sigma)
                
                artmp = (arx[:mu] - xold) / sigma
                diagC = (1 - c1 - cmu_val) * diagC + \
                        c1 * (pc**2 + (1-hsig)*cc*(2-cc)*diagC) + \
                        cmu_val * np.sum(weights[:, None] * artmp**2, axis=0)
                diagC = np.maximum(diagC, 1e-20)
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ ((xmean - xold) / sigma))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*local_evals/lam)) / chiN) < (1.4 + 2/(n+1))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ((xmean - xold) / sigma)
                
                artmp = (arx[:mu] - xold) / sigma
                C = (1 - c1 - cmu_val) * C + \
                    c1 * (np.outer(pc, pc) + (1-hsig)*cc*(2-cc)*C) + \
                    cmu_val * (artmp.T @ np.diag(weights) @ artmp)
            
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, 2 * max(ranges))
            
            if not use_sep and local_evals - eigeneval > lam / (c1 + cmu_val + 1e-30) / n / 10:
                eigeneval = local_evals
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.maximum(D_sq, 1e-20)
                    D = np.sqrt(D_sq)
                    invsqrtC = B @ np.diag(1.0/D) @ B.T
                except:
                    break
            
            if use_sep:
                cond = np.max(diagC) / (np.min(diagC) + 1e-30)
                size_metric = sigma * np.sqrt(np.max(diagC))
            else:
                cond = np.max(D)**2 / (np.min(D)**2 + 1e-30)
                size_metric = sigma * np.max(D)
            
            if size_metric < 1e-14 * max(ranges) or stag_count > 20 + 5*n or cond > 1e14:
                break

    # ---- L-SHADE ----
    def run_lshade(max_t, pop_mult=1.0):
        nonlocal best, best_x
        pop_size_init = min(max(int(60 * pop_mult), 18 * dim), 800)
        pop_size = pop_size_init
        min_pop = 4
        
        # Initialize with LHS + opposition
        half = pop_size_init // 2
        base_pop = lhs_sample(half)
        opp_pop = lower + upper - base_pop
        init_pop = np.vstack([base_pop, opp_pop])
        if len(init_pop) > pop_size_init:
            init_pop = init_pop[:pop_size_init]
        elif len(init_pop) < pop_size_init:
            extra = lhs_sample(pop_size_init - len(init_pop))
            init_pop = np.vstack([init_pop, extra])
        
        population = init_pop
        fitness = np.array([eval_f(ind) for ind in population])
        
        memory_size = max(6, dim // 2)
        memory_size = min(memory_size, 20)
        M_F = np.full(memory_size, 0.5)
        M_CR = np.full(memory_size, 0.8)
        mem_idx = 0
        archive = []
        archive_max = pop_size_init
        t_end = elapsed() + max_t
        gen = 0
        total_evals_shade = 0
        max_evals_shade = pop_size_init * 200  # rough estimate
        
        while elapsed() < t_end and time_left() > 0.3:
            gen += 1
            S_F, S_CR, delta_f = [], [], []
            
            # Adaptive p-best rate
            progress = min(1.0, total_evals_shade / (max_evals_shade + 1))
            p_best_rate = max(2.0/pop_size, 0.25 - 0.20 * progress)  # from 0.25 to 0.05
            
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            sorted_idx = np.argsort(fitness)
            
            for i in range(pop_size):
                if elapsed() >= t_end or time_left() < 0.2:
                    break
                
                ri = np.random.randint(0, memory_size)
                
                # Generate Fi from Cauchy
                Fi = -1
                attempts = 0
                while Fi <= 0 and attempts < 10:
                    Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                    attempts += 1
                if Fi <= 0:
                    Fi = 0.01
                Fi = min(Fi, 1.0)
                
                # Generate CRi from Normal
                if M_CR[ri] < 0:
                    CRi = 0.0
                else:
                    CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
                
                # p-best
                p = max(2, int(p_best_rate * pop_size))
                pbest = sorted_idx[np.random.randint(0, p)]
                
                # r1 different from i
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(pop_size)
                
                # r2 from pop+archive, different from i and r1
                total_pool = pop_size + len(archive)
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(total_pool)
                
                if r2 < pop_size:
                    xr2 = population[r2]
                else:
                    xr2 = archive[r2 - pop_size]
                
                # current-to-pbest/1
                mutant = population[i] + Fi * (population[pbest] - population[i]) + Fi * (population[r1] - xr2)
                
                # Binomial crossover
                cross = np.random.rand(dim) < CRi
                cross[np.random.randint(dim)] = True
                trial = np.where(cross, mutant, population[i])
                
                # Bounds handling - midpoint
                for d in range(dim):
                    if trial[d] < lower[d]:
                        trial[d] = (lower[d] + population[i][d]) / 2
                    elif trial[d] > upper[d]:
                        trial[d] = (upper[d] + population[i][d]) / 2
                
                tf = eval_f(trial)
                total_evals_shade += 1
                
                if tf <= fitness[i]:
                    if tf < fitness[i]:
                        S_F.append(Fi)
                        S_CR.append(CRi)
                        delta_f.append(abs(fitness[i] - tf))
                        archive.append(population[i].copy())
                        if len(archive) > archive_max:
                            archive.pop(np.random.randint(len(archive)))
                    new_pop[i] = trial
                    new_fit[i] = tf
            
            population = new_pop
            fitness = new_fit
            
            # Update memory
            if S_F:
                w = np.array(delta_f)
                w = w / (w.sum() + 1e-30)
                sf = np.array(S_F)
                scr = np.array(S_CR)
                # Weighted Lehmer mean for F
                M_F[mem_idx] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
                # Weighted mean for CR
                M_CR[mem_idx] = np.sum(w * scr)
                mem_idx = (mem_idx + 1) % memory_size
            
            # Linear population size reduction
            progress_time = (elapsed() - (t_end - max_t)) / max_t if max_t > 0 else 1.0
            new_ps = max(min_pop, int(round(pop_size_init - (pop_size_init - min_pop) * progress_time)))
            if new_ps < pop_size:
                sidx = np.argsort(fitness)
                population = population[sidx[:new_ps]]
                fitness = fitness[sidx[:new_ps]]
                pop_size = new_ps
                sorted_idx = np.argsort(fitness)

    # ---- Pattern Search ----
    def pattern_search(x0, max_t, init_step=0.1):
        nonlocal best, best_x
        t_end = elapsed() + max_t
        x = x0.copy()
        fx = eval_f(x)
        step = ranges * init_step
        
        while elapsed() < t_end and time_left() > 0.02:
            improved = False
            for
