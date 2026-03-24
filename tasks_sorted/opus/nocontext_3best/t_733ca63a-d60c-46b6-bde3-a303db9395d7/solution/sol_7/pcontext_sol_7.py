#
#
#Looking at the results, algorithm #1 (score 1.74) performed best with L-SHADE + Nelder-Mead + coordinate search. I'll improve it by:
#
#1. Better time allocation - more balanced between global and local search
#2. Improved SHADE with weighted Lehmer mean
#3. More aggressive and frequent local search with multiple restarts
#4. Adding a golden section line search along promising directions
#5. Better restart detection and diversification
#6. Final intensive multi-method local search phase
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_solution = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    rng = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_ok(fraction=0.96):
        return elapsed() < max_time * fraction
    
    def ev(x):
        nonlocal best, best_solution
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_solution = x.copy()
        return f
    
    def opposite(x):
        return np.clip(lower + upper - x, lower, upper)
    
    # LHS initialization
    init_pop_size = min(max(14 * dim, 50), 200)
    min_pop_size = max(4, dim)
    
    pop = np.random.uniform(0, 1, (init_pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(init_pop_size)
        pop[:, i] = (perm + pop[:, i]) / init_pop_size
    pop = lower + pop * rng
    
    fit = np.full(init_pop_size, float('inf'))
    for i in range(init_pop_size):
        if not time_ok():
            return best
        fit[i] = ev(pop[i])
        if time_ok() and i < init_pop_size // 2:
            opp = opposite(pop[i])
            of = ev(opp)
            if of < fit[i]:
                pop[i] = opp
                fit[i] = of
    
    # SHADE memory
    H = 10
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.8)
    mi = 0
    archive = []
    
    stagnation = 0
    prev_best = best
    generation = 0
    
    # Nelder-Mead local search
    def nelder_mead(x0, max_evals_nm, initial_scale=0.05, time_frac=0.93):
        n = len(x0)
        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i + 1] = x0.copy()
            delta = initial_scale * rng[i]
            simplex[i + 1][i] += delta if np.random.random() > 0.5 else -delta
            simplex[i + 1] = np.clip(simplex[i + 1], lower, upper)
        
        f_simplex = np.zeros(n + 1)
        for i in range(n + 1):
            if not time_ok(time_frac):
                return
            f_simplex[i] = ev(simplex[i])
        
        used = n + 1
        no_improve = 0
        best_local = f_simplex.min()
        
        while used < max_evals_nm and time_ok(time_frac):
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            if f_simplex[0] < best_local - 1e-15:
                best_local = f_simplex[0]
                no_improve = 0
            else:
                no_improve += 1
            if no_improve > max(n, 20):
                break
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, lower, upper)
            fr = ev(xr); used += 1
            
            if fr < f_simplex[0]:
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, lower, upper)
                fe = ev(xe); used += 1
                if fe < fr:
                    simplex[-1] = xe; f_simplex[-1] = fe
                else:
                    simplex[-1] = xr; f_simplex[-1] = fr
            elif fr < f_simplex[-2]:
                simplex[-1] = xr; f_simplex[-1] = fr
            else:
                if fr < f_simplex[-1]:
                    xc = centroid + rho * (xr - centroid)
                    xc = np.clip(xc, lower, upper)
                    fc = ev(xc); used += 1
                    if fc <= fr:
                        simplex[-1] = xc; f_simplex[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            if not time_ok(time_frac): return
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            f_simplex[i] = ev(simplex[i]); used += 1
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, lower, upper)
                    fc = ev(xc); used += 1
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc; f_simplex[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            if not time_ok(time_frac): return
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            f_simplex[i] = ev(simplex[i]); used += 1
            
            spread = np.max(np.abs(simplex[-1] - simplex[0]) / np.maximum(rng, 1e-30))
            if spread < 1e-16:
                break
    
    # Coordinate search with acceleration
    def coord_search(x0, max_evals_cs, step_scale=0.01, time_frac=0.93):
        x = x0.copy()
        fx = ev(x)
        used = 1
        step = step_scale * rng
        
        improved = True
        while improved and used < max_evals_cs and time_ok(time_frac):
            improved = False
            for j in range(dim):
                if not time_ok(time_frac) or used >= max_evals_cs:
                    return
                for sign in [1.0, -1.0]:
                    xp = x.copy()
                    xp[j] = np.clip(x[j] + sign * step[j], lower[j], upper[j])
                    fp = ev(xp); used += 1
                    if fp < fx:
                        x = xp; fx = fp; improved = True
                        while used < max_evals_cs and time_ok(time_frac):
                            step[j] *= 1.5
                            xp2 = x.copy()
                            xp2[j] = np.clip(x[j] + sign * step[j], lower[j], upper[j])
                            fp2 = ev(xp2); used += 1
                            if fp2 < fx:
                                x = xp2; fx = fp2
                            else:
                                step[j] /= 1.5
                                break
                        break
            step *= 0.5
            if np.max(step / rng) < 1e-16:
                break
    
    # Pattern search (Hooke-Jeeves)
    def pattern_search(x0, max_evals, step_scale=0.02, time_frac=0.93):
        x = x0.copy()
        fx = ev(x)
        used = 1
        step = step_scale * rng
        
        while used < max_evals and time_ok(time_frac) and np.max(step/rng) > 1e-16:
            x_base = x.copy()
            fx_base = fx
            # Exploratory moves
            for j in range(dim):
                if not time_ok(time_frac) or used >= max_evals:
                    return
                xp = x.copy(); xp[j] = np.clip(x[j] + step[j], lower[j], upper[j])
                fp = ev(xp); used += 1
                if fp < fx:
                    x = xp; fx = fp
                else:
                    xn = x.copy(); xn[j] = np.clip(x[j] - step[j], lower[j], upper[j])
                    fn = ev(xn); used += 1
                    if fn < fx:
                        x = xn; fx = fn
            
            if fx < fx_base:
                # Pattern move
                direction = x - x_base
                xp = np.clip(x + direction, lower, upper)
                fp = ev(xp); used += 1
                if fp < fx:
                    x = xp; fx = fp
            else:
                step *= 0.5
    
    # Line search along a direction
    def line_search(x0, direction, max_evals=20, time_frac=0.93):
        x = x0.copy()
        fx = ev(x)
        used = 1
        d_norm = np.linalg.norm(direction)
        if d_norm < 1e-30:
            return
        d = direction / d_norm
        
        # Try multiple step sizes
        best_alpha = 0.0
        scale = np.mean(rng) * 0.01
        for mult in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, -0.001, -0.005, -0.01, -0.02, -0.05, -0.1, -0.2, -0.5, -1.0]:
            if not time_ok(time_frac) or used >= max_evals:
                return
            alpha = mult * scale * d_norm
            xt = np.clip(x + alpha * d, lower, upper)
            ft = ev(xt); used += 1
            if ft < fx:
                fx = ft
                best_alpha = alpha
                x = xt
    
    # === Main SHADE loop ===
    global_phase_end = 0.55  # Use 55% of time for global search
    
    while time_ok(global_phase_end):
        generation += 1
        S_F, S_CR, S_d = [], [], []
        
        sorted_idx = np.argsort(fit)
        current_pop_size = len(pop)
        
        for i in range(current_pop_size):
            if not time_ok(global_phase_end):
                break
            
            ri = np.random.randint(H)
            while True:
                Fi = M_F[ri] + 0.1 * np.random.standard_cauchy()
                if Fi > 0: break
            Fi = min(Fi, 1.0)
            CRi = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0, 1)
            
            time_ratio = elapsed() / max_time
            
            p = max(2, int(max(0.05, 0.2 - 0.15 * time_ratio) * current_pop_size))
            pb = sorted_idx[np.random.randint(p)]
            
            idxs = [j for j in range(current_pop_size) if j != i]
            r1 = np.random.choice(idxs)
            
            pool_size = current_pop_size + len(archive)
            r2 = np.random.randint(pool_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(pool_size)
            xr2 = pop[r2] if r2 < current_pop_size else archive[r2 - current_pop_size]
            
            mutant = pop[i] + Fi * (pop[pb] - pop[i]) + Fi * (pop[r1] - xr2)
            
            trial = pop[i].copy()
            jr = np.random.randint(dim)
            mask = np.random.random(dim) < CRi
            mask[jr] = True
            trial[mask] = mutant[mask]
            
            below = trial < lower; above = trial > upper
            trial[below] = (lower[below] + pop[i][below]) / 2
            trial[above] = (upper[above] + pop[i][above]) / 2
            
            tf = ev(trial)
            if tf <= fit[i]:
                delta = fit[i] - tf
                if delta > 0:
                    S_F.append(Fi); S_CR.append(CRi); S_d.append(delta)
                if len(archive) < init_pop_size:
                    archive.append(pop[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = pop[i].copy()
                pop[i] = trial; fit[i] = tf
        
        if S_F:
            w = np.array(S_d); w /= (w.sum() + 1e-30)
            sf = np.array(S_F)
            M_F[mi] = np.sum(w * sf**2) / max(np.sum(w * sf), 1e-30)
            M_CR[mi] = np.sum(w * np.array(S_CR))
            mi = (mi + 1) % H
        
        # L-SHADE population reduction
        time_ratio = elapsed() / max_time
        target_ratio = min(time_ratio / global_phase_end, 1.0)
        new_pop_size = max(min_pop_size, int(round(init_pop_size - (init_pop_size - min_pop_size) * target_ratio)))
        if new_pop_size < len(pop):
            si = np.argsort(fit)
            pop = pop[si[:new_pop_size]]
            fit = fit[si[:new_pop_size]]
            while len(archive) > new_pop_size:
                archive.pop(np.random.randint(len(archive)))
        
        if abs(prev_best - best) < 1e-15:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # Periodic local search during global phase
        if stagnation > 0 and stagnation % 5 == 0 and time_ok(0.50) and best_solution is not None:
            scale = 0
