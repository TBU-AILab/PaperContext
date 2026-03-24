#Looking at the previous algorithm, I see it scored 68.01. Let me analyze potential improvements:
#
#1. **Better time allocation** - CMA-ES restarts may not get enough budget each
#2. **Differential Evolution** could help for multimodal landscapes before CMA-ES refinement
#3. **Adaptive restart strategy** - smarter selection of restart points
#4. **Better local search** - Nelder-Mead with adaptive simplex size
#
#Let me redesign with a more balanced hybrid approach:
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
    
    def remaining():
        return max(0, max_time - elapsed())
    
    def eval_f(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    def rand_in_bounds():
        return lower + np.random.rand(dim) * ranges

    # === Phase 1: DE/current-to-pbest with archive (JADE-like) ===
    NP = max(min(10 * dim, 200), 40)
    pop = np.array([rand_in_bounds() for _ in range(NP)])
    fit = np.array([eval_f(p) for p in pop])
    
    archive = []
    mu_F = 0.5
    mu_CR = 0.5
    
    de_deadline = max_time * 0.55
    
    gen = 0
    while elapsed() < de_deadline:
        gen += 1
        S_F, S_CR = [], []
        new_pop = pop.copy()
        new_fit = fit.copy()
        
        p_best_rate = max(0.05, 0.2 - 0.15 * elapsed() / de_deadline)
        
        for i in range(NP):
            if elapsed() >= de_deadline:
                break
            
            # Generate F and CR
            Fi = np.clip(np.random.standard_cauchy() * 0.1 + mu_F, 0.01, 1.0)
            CRi = np.clip(np.random.normal(mu_CR, 0.1), 0.0, 1.0)
            
            # p-best
            p_size = max(2, int(NP * p_best_rate))
            p_best_idx = np.argsort(fit)[:p_size]
            x_pbest = pop[np.random.choice(p_best_idx)]
            
            # r1 != i
            candidates = list(range(NP))
            candidates.remove(i)
            r1 = np.random.choice(candidates)
            
            # r2 from pop + archive, != i, r1
            pool = list(range(NP)) + list(range(NP, NP + len(archive)))
            pool = [x for x in pool if x != i and x != r1]
            r2_idx = np.random.choice(pool)
            if r2_idx < NP:
                x_r2 = pop[r2_idx]
            else:
                x_r2 = archive[r2_idx - NP]
            
            # Mutation: current-to-pbest/1
            mutant = pop[i] + Fi * (x_pbest - pop[i]) + Fi * (pop[r1] - x_r2)
            
            # Bounce-back
            for d in range(dim):
                if mutant[d] < lower[d]:
                    mutant[d] = lower[d] + np.random.rand() * (pop[i][d] - lower[d])
                elif mutant[d] > upper[d]:
                    mutant[d] = upper[d] - np.random.rand() * (upper[d] - pop[i][d])
            mutant = np.clip(mutant, lower, upper)
            
            # Crossover
            trial = pop[i].copy()
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < CRi
            mask[j_rand] = True
            trial[mask] = mutant[mask]
            
            f_trial = eval_f(trial)
            
            if f_trial <= fit[i]:
                if f_trial < fit[i]:
                    archive.append(pop[i].copy())
                    S_F.append(Fi)
                    S_CR.append(CRi)
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        pop = new_pop
        fit = new_fit
        
        # Trim archive
        while len(archive) > NP:
            archive.pop(np.random.randint(len(archive)))
        
        # Adapt mu_F, mu_CR (Lehmer mean for F)
        if S_F:
            weights_s = np.array(S_F)
            mu_F = (1 - 0.1) * mu_F + 0.1 * np.sum(np.array(S_F)**2) / (np.sum(np.array(S_F)) + 1e-30)
            mu_CR = (1 - 0.1) * mu_CR + 0.1 * np.mean(S_CR)
        
        # Population size reduction (L-SHADE)
        min_NP = max(4, dim)
        new_NP = max(min_NP, int(round(NP - (NP - min_NP) * elapsed() / de_deadline)))
        if new_NP < len(pop):
            order = np.argsort(fit)[:new_NP]
            pop = pop[order]
            fit = fit[order]
            NP = new_NP

    # Collect diverse elite seeds for CMA-ES
    elite_pool = [(best, best_params.copy())]
    sorted_idx = np.argsort(fit)
    for idx in sorted_idx[:5]:
        if fit[idx] < best * 5 + 1e-10:
            duplicate = False
            for _, ep in elite_pool:
                if np.max(np.abs(pop[idx] - ep) / (ranges + 1e-30)) < 0.01:
                    duplicate = True
                    break
            if not duplicate:
                elite_pool.append((fit[idx], pop[idx].copy()))

    # === Phase 2: CMA-ES restarts ===
    def run_cmaes(init_mean, init_sigma, deadline, lam_override=None):
        nonlocal best, best_params
        n = dim
        lam = lam_override if lam_override else max(4 + int(3 * np.log(n)), 10)
        mu = lam // 2
        
        w_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = w_raw / w_raw.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n+1.3)**2 + mu_eff)
        cmu_val = min(1 - c1, 2*(mu_eff - 2 + 1/mu_eff) / ((n+2)**2 + mu_eff))
        damps = 1 + 2*max(0, np.sqrt((mu_eff-1)/(n+1)) - 1) + cs
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        mean = np.clip(init_mean.copy(), lower, upper)
        sigma = init_sigma
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        use_full = (n <= 100)
        
        if use_full:
            C = np.eye(n)
            eig_vals_sq = np.ones(n)
            B = np.eye(n)
            invsqrtC = np.eye(n)
            eigen_counter = 0
            eigen_interval = max(1, int(1 / (10*n*(c1 + cmu_val) + 1e-30)))
        else:
            diag_C = np.ones(n)
        
        stale = 0
        best_gen_f = float('inf')
        f_hist = []
        g = 0
        
        while elapsed() < deadline:
            g += 1
            offspring = np.empty((lam, n))
            f_off = np.empty(lam)
            
            for i in range(lam):
                if elapsed() >= deadline:
                    return
                z = np.random.randn(n)
                if use_full:
                    offspring[i] = mean + sigma * (B @ (eig_vals_sq * z))
                else:
                    offspring[i] = mean + sigma * np.sqrt(np.maximum(diag_C, 1e-20)) * z
                offspring[i] = np.clip(offspring[i], lower, upper)
                f_off[i] = eval_f(offspring[i])
            
            order = np.argsort(f_off)
            selected = offspring[order[:mu]]
            
            old_mean = mean.copy()
            mean = np.dot(weights, selected)
            mean = np.clip(mean, lower, upper)
            
            diff = (mean - old_mean) / max(sigma, 1e-30)
            
            if use_full:
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mu_eff) * (invsqrtC @ diff)
            else:
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mu_eff) * diff / np.sqrt(np.maximum(diag_C, 1e-20))
            
            ps_norm = np.linalg.norm(ps)
            hs = 1 if ps_norm / np.sqrt(max(1 - (1-cs)**(2*g), 1e-30)) < (1.4 + 2/(n+1)) * chi_n else 0
            
            pc = (1-cc)*pc + hs * np.sqrt(cc*(2-cc)*mu_eff) * diff
            
            if use_full:
                artmp = ((selected - old_mean) / max(sigma, 1e-30)).T
                C = (1 - c1 - cmu_val + (1-hs)*c1*cc*(2-cc)) * C + \
                    c1 * np.outer(pc, pc) + cmu_val * (artmp @ np.diag(weights) @ artmp.T)
                C = np.triu(C) + np.triu(C, 1).T
                
                eigen_counter += 1
                if eigen_counter >= eigen_interval:
                    eigen_counter = 0
                    try:
                        ev, B = np.linalg.eigh(C)
                        ev = np.maximum(ev, 1e-20)
                        eig_vals_sq = np.sqrt(ev)
                        invsqrtC = B @ np.diag(1.0/eig_vals_sq) @ B.T
                    except:
                        C = np.eye(n); eig_vals_sq = np.ones(n)
                        B = np.eye(n); invsqrtC = np.eye(n)
            else:
                artmp = (selected - old_mean) / max(sigma, 1e-30)
                diag_C = (1 - c1 - cmu_val + (1-hs)*c1*cc*(2-cc)) * diag_C + \
                         c1 * pc**2 + cmu_val * np.sum(weights[:, None] * artmp**2, axis=0)
                diag_C = np.maximum(diag_C, 1e-20)
            
            sigma *= np.exp(np.clip((cs/damps) * (ps_norm/chi_n - 1), -0.5, 0.5))
            sigma = np.clip(sigma, 1e-17 * np.mean(ranges), 2.0 * np.mean(ranges))
            
            cur_best = f_off[order[0]]
            f_hist.append(cur_best)
            
            if cur_best < best_gen_f - 1e-15:
                best_gen_f = cur_best
                stale = 0
            else:
                stale += 1
            
            if use_full:
                max_std = sigma * np.max(eig_vals_sq)
            else:
                max_std = sigma * np.max(np.sqrt(diag_C))
            
            if max_std < 1e-14 * np.mean(ranges):
                return
            if stale > 15 + 30 * n // lam:
                return
            if len(f_hist) > 50:
                recent = f_hist[-50:]
                if max(recent) - min(recent) < 1e-14 * (abs(best) + 1e-30):
                    return

    base_lam = max(4 + int(3 * np.log(dim)), 10)
    restart = 0
    lam_mult = 1.0
    
    while elapsed() < max_time * 0.88:
        if remaining() < 0.3:
            break
        
        if restart < len(elite_pool):
            _, sp = elite_pool[restart]
            sig = 0.15 * np.mean(ranges)
            lam = base_lam
        elif restart == len(elite_pool):
            sp = best_params.copy()
            sig = 0.01 * np.mean(ranges)
            lam = base_lam
        elif restart % 4 == 0:
            lam_mult = min(lam_mult * 2, 10)
            sp = rand_in_bounds()
            sig = 0.3 * np.mean(ranges)
            lam = int(base_lam * lam_mult)
        elif restart % 4 == 1:
            sp = best_params + 0.05 * ranges * np.random.randn(dim)
            sp = np.clip(sp, lower, upper)
            sig = 0.05 * np.mean(ranges)
            lam = base_lam
        elif restart % 4 == 2:
            sp = rand_in_bounds()
            sig = 0.2 * np.mean(ranges)
            lam = base_lam
        else:
            sp = best_params + 0.002 * ranges * np.random.randn(dim)
            sp = np.clip(sp, lower, upper)
            sig = 0.005 * np.mean(ranges)
            lam = base_lam
        
        budget = min(0.25, max(0.05, remaining() * 0.25))
        deadline = elapsed() + budget
        deadline = min(deadline, max_time * 0.88)
        
        run_cmaes(sp, sig, deadline, lam)
        restart += 1

    # === Phase 3: Coordinate descent with golden section ===
    if best_params is not None and elapsed() < max_time * 0.95:
        x = best_params.copy()
        fx = best
        step = 0.001 * ranges.copy()
        
        for _ in range(5):
            if elapsed() >= max_time * 0.95:
                break
            improved = False
            perm = np.random.permutation(dim)
            for d in perm:
                if elapsed() >= max_time * 0.95:
                    break
                for sign in [1, -1]:
                    xn = x.copy()
                    xn[d] = np.clip(x[d] + sign * step[d], lower[d], upper[d])
                    fn = eval_f(xn)
                    if fn < fx:
                        while elapsed() < max_time * 0.95:
                            step[d] *= 1.5
                            xn2 = x.copy()
                            xn2[d] = np.clip(xn[d] + sign * step[d], lower[d], upper[d])
                            fn2 = eval_f(xn2)
                            if fn2 < fn:
                                xn, fn = xn2, fn2
                            else:
                                break
                        x, fx = xn, fn
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if np.max(step / ranges) < 1e-15:
                    break

    # === Phase 4: Final perturbations ===
    while elapsed() < max_time * 0.999:
        sc = 10 ** np.random.uniform(-7, -3)
        candidate = best_params + sc * ranges * np.random.randn(dim)
        candidate = np.clip(candidate, lower, upper)
        eval_f(candidate)
    
    return best
#
#Key changes:
#1. **L-SHADE DE** as primary explorer (55% time) with population size reduction for efficiency
#2. **Diverse elite seeding** - extracts multiple diverse good solutions from DE for CMA-ES restarts
#3. **Better CMA-ES restart scheduling** - seeds from elite pool first, then alternates between IPOP and local refinement
#4. **Bounce-back boundary handling** in DE instead of simple clipping
#5. **Better time balancing** between exploration (DE) and exploitation (CMA-ES + local search)
