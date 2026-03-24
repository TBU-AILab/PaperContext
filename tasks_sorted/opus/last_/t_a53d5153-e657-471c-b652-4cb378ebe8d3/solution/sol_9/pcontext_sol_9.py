#Looking at the previous algorithm, it scored 24.97. Let me analyze potential improvements:
#
#1. **L-SHADE DE portion could be more efficient** - spending 55% on DE might be too much
#2. **CMA-ES restarts need better strategy** - more focused local search around best
#3. **Add Nelder-Mead simplex** for polishing
#4. **Better adaptive parameters** and more aggressive exploitation near the end
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
    
    evals = [0]
    
    def eval_f(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        evals[0] += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f

    def rand_in_bounds():
        return lower + np.random.rand(dim) * ranges

    # === Phase 1: L-SHADE DE ===
    NP_init = max(min(18 * dim, 300), 50)
    NP = NP_init
    pop = np.array([rand_in_bounds() for _ in range(NP)])
    fit = np.array([eval_f(p) for p in pop])
    
    archive = []
    mu_F = 0.5
    mu_CR = 0.5
    H = 6  # memory size
    M_F = [0.5] * H
    M_CR = [0.5] * H
    k = 0
    
    de_deadline = max_time * 0.45
    
    while elapsed() < de_deadline:
        S_F, S_CR, S_df = [], [], []
        new_pop = pop.copy()
        new_fit = fit.copy()
        
        progress = elapsed() / de_deadline
        p_best_rate = max(0.05, 0.25 - 0.20 * progress)
        
        for i in range(len(pop)):
            if elapsed() >= de_deadline:
                break
            
            ri = np.random.randint(H)
            Fi = -1
            while Fi <= 0:
                Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            p_size = max(2, int(len(pop) * p_best_rate))
            p_best_idx = np.argsort(fit)[:p_size]
            x_pbest = pop[np.random.choice(p_best_idx)]
            
            candidates = list(range(len(pop)))
            candidates.remove(i)
            r1 = np.random.choice(candidates)
            
            union_size = len(pop) + len(archive)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(union_size)
            
            if r2 < len(pop):
                x_r2 = pop[r2]
            else:
                x_r2 = archive[r2 - len(pop)]
            
            mutant = pop[i] + Fi * (x_pbest - pop[i]) + Fi * (pop[r1] - x_r2)
            
            for d in range(dim):
                if mutant[d] < lower[d]:
                    mutant[d] = (lower[d] + pop[i][d]) / 2.0
                elif mutant[d] > upper[d]:
                    mutant[d] = (upper[d] + pop[i][d]) / 2.0
            
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
                    S_df.append(abs(fit[i] - f_trial))
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        pop = new_pop
        fit = new_fit
        
        while len(archive) > len(pop):
            archive.pop(np.random.randint(len(archive)))
        
        if S_F:
            w = np.array(S_df)
            w = w / (w.sum() + 1e-30)
            M_F[k] = float(np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30))
            M_CR[k] = float(np.sum(w * np.array(S_CR)))
            k = (k + 1) % H
        
        min_NP = max(4, dim)
        new_NP = max(min_NP, int(round(NP_init + (min_NP - NP_init) * elapsed() / de_deadline)))
        if new_NP < len(pop):
            order = np.argsort(fit)[:new_NP]
            pop = pop[order]
            fit = fit[order]

    elite_pool = []
    sorted_idx = np.argsort(fit)
    for idx in sorted_idx[:min(10, len(pop))]:
        duplicate = False
        for _, ep in elite_pool:
            if np.max(np.abs(pop[idx] - ep) / (ranges + 1e-30)) < 0.02:
                duplicate = True
                break
        if not duplicate:
            elite_pool.append((fit[idx], pop[idx].copy()))
    if best_params is not None:
        elite_pool.insert(0, (best, best_params.copy()))

    # === Phase 2: CMA-ES restarts ===
    def run_cmaes(init_mean, init_sigma, deadline, lam_override=None):
        nonlocal best, best_params
        n = dim
        lam = lam_override or max(4 + int(3 * np.log(n)), 12)
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
        
        use_full = n <= 80
        
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
                C = (1 - c1 - cmu_val + (1-hs)*c1*cc*(2-cc)) * C + c1 * np.outer(pc, pc) + cmu_val * (artmp @ np.diag(weights) @ artmp.T)
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
                        C = np.eye(n); eig_vals_sq = np.ones(n); B = np.eye(n); invsqrtC = np.eye(n)
            else:
                artmp = (selected - old_mean) / max(sigma, 1e-30)
                diag_C = (1 - c1 - cmu_val + (1-hs)*c1*cc*(2-cc)) * diag_C + c1 * pc**2 + cmu_val * np.sum(weights[:, None] * artmp**2, axis=0)
                diag_C = np.maximum(diag_C, 1e-20)
            
            sigma *= np.exp(np.clip((cs/damps) * (ps_norm/chi_n - 1), -0.5, 0.5))
            sigma = np.clip(sigma, 1e-17 * np.mean(ranges), 2.0 * np.mean(ranges))
            
            cur_best = f_off[order[0]]
            if cur_best < best_gen_f - 1e-15:
                best_gen_f = cur_best
                stale = 0
            else:
                stale += 1
            
            if use_full:
                max_std = sigma * np.max(eig_vals_sq)
            else:
                max_std = sigma * np.max(np.sqrt(diag_C))
            
            if max_std < 1e-15 * np.mean(ranges):
                return
            if stale > 20 + 30 * n // lam:
                return

    base_lam = max(4 + int(3 * np.log(dim)), 12)
    restart = 0
    lam_mult = 1.0
    
    while elapsed() < max_time * 0.90:
        if remaining() < 0.2:
            break
        
        if restart < len(elite_pool):
            _, sp = elite_pool[restart]
            sig = 0.1 * np.mean(ranges)
            lam = base_lam
        elif restart % 5 == 0:
            sp = best_params + 0.001 * ranges * np.random.randn(dim)
            sp = np.clip(sp, lower, upper)
            sig = 0.003 * np.mean(ranges)
            lam = base_lam
        elif restart % 5 == 1:
            lam_mult = min(lam_mult * 2, 8)
            sp = rand_in_bounds()
            sig = 0.3 * np.mean(ranges)
            lam = int(base_lam * lam_mult)
        elif restart % 5 == 2:
            sp = best_params + 0.05 * ranges * np.random.randn(dim)
            sp = np.clip(sp, lower, upper)
            sig = 0.03 * np.mean(ranges)
            lam = base_lam
        elif restart % 5 == 3:
            sp = rand_in_bounds()
            sig = 0.2 * np.mean(ranges)
            lam = base_lam
        else:
            sp = best_params + 0.0005 * ranges * np.random.randn(dim)
            sp = np.clip(sp, lower, upper)
            sig = 0.001 * np.mean(ranges)
            lam = base_lam
        
        budget_t = min(remaining() * 0.3, max(0.3, remaining() * 0.15))
        deadline = min(elapsed() + budget_t, max_time * 0.90)
        
        run_cmaes(sp, sig, deadline, lam)
        restart += 1

    # === Phase 3: Fine local search ===
    if best_params is not None:
        x = best_params.copy()
        fx = best
        step = 0.0005 * ranges.copy()
        
        for _ in range(10):
            if elapsed() >= max_time * 0.97:
                break
            improved = False
            for d in np.random.permutation(dim):
                if elapsed() >= max_time * 0.97:
                    break
                for sign in [1, -1]:
                    xn = x.copy()
                    xn[d] = np.clip(x[d] + sign * step[d], lower[d], upper[d])
                    fn = eval_f(xn)
                    if fn < fx:
                        while elapsed() < max_time * 0.97:
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
                if np.max(step / ranges) < 1e-16:
                    break

    while elapsed() < max_time * 0.999:
        sc = 10 ** np.random.uniform(-8, -4)
        candidate = best_params + sc * ranges * np.random.randn(dim)
        candidate = np.clip(candidate, lower, upper)
        eval_f(candidate)
    
    return best
