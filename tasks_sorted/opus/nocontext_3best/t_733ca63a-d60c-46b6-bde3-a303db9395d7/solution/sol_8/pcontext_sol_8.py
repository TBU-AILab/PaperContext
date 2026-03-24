#Looking at the best algorithm (score 1.74), I'll improve it with:
#
#1. Better time management with dedicated phases
#2. CMA-ES-inspired local search for better exploitation
#3. More sophisticated restart with diverse strategies
#4. Gradient approximation for direction-guided search
#5. Multi-start local search from elite solutions
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
    
    # Elite tracking
    elites = []
    elite_max = 8
    def update_elite(x, f):
        nonlocal elites
        for i, (ef, ex) in enumerate(elites):
            if np.linalg.norm(x - ex) < 0.01 * np.linalg.norm(rng):
                if f < ef:
                    elites[i] = (f, x.copy())
                return
        elites.append((f, x.copy()))
        elites.sort(key=lambda t: t[0])
        if len(elites) > elite_max:
            elites = elites[:elite_max]
    
    # LHS initialization
    init_pop_size = min(max(12 * dim, 50), 180)
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
        update_elite(pop[i], fit[i])
        if time_ok() and i < init_pop_size // 3:
            opp = opposite(pop[i])
            of = ev(opp)
            if of < fit[i]:
                pop[i] = opp
                fit[i] = of
            update_elite(pop[i], fit[i])
    
    # SHADE memory
    H = 12
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.8)
    mi = 0
    archive = []
    
    stagnation = 0
    prev_best = best
    generation = 0
    
    # Nelder-Mead
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
            if not time_ok(time_frac): return
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
    
    # Approximate gradient descent with momentum
    def gradient_search(x0, max_evals, step_scale=0.005, time_frac=0.93):
        x = x0.copy()
        fx = ev(x)
        used = 1
        velocity = np.zeros(dim)
        momentum = 0.7
        
        for iteration in range(max_evals // (2 * dim + 2)):
            if not time_ok(time_frac) or used >= max_evals:
                return
            # Finite difference gradient
            grad = np.zeros(dim)
            h = step_scale * rng
            for j in range(dim):
                if not time_ok(time_frac) or used + 2 > max_evals:
                    return
                xp = x.copy(); xp[j] = np.clip(x[j] + h[j], lower[j], upper[j])
                xn = x.copy(); xn[j] = np.clip(x[j] - h[j], lower[j], upper[j])
                fp = ev(xp); fn = ev(xn); used += 2
                grad[j] = (fp - fn) / (2 * h[j] + 1e-30)
            
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-15:
                break
            
            velocity = momentum * velocity - step_scale * np.mean(rng) * grad / gnorm
            x_new = np.clip(x + velocity, lower, upper)
            fx_new = ev(x_new); used += 1
            
            if fx_new < fx:
                x = x_new; fx = fx_new
                step_scale *= 1.1
            else:
                velocity *= 0.0
                step_scale *= 0.5
                if step_scale < 1e-15:
                    break
    
    # CMA-ES inspired local search
    def cma_local(x0, sigma0=0.05, max_evals=500, time_frac=0.93):
        n = dim
        lam = max(4 + int(3 * np.log(n)), 8)
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu_val = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        xmean = x0.copy()
        sigma = sigma0 * np.mean(rng)
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        used = 0
        
        for gen in range(max_evals // lam):
            if not time_ok(time_frac) or used + lam > max_evals:
                return
            
            try:
                sqrtC = np.linalg.cholesky(C)
            except:
                C = np.eye(n)
                sqrtC = np.eye(n)
            
            arz = np.random.randn(lam, n)
            arx = np.zeros((lam, n))
            arf = np.zeros(lam)
            
            for k in range(lam):
                if not time_ok(time_frac):
                    return
                arx[k] = np.clip(xmean + sigma * sqrtC @ arz[k], lower, upper)
                arf[k] = ev(arx[k])
                used += 1
            
            idx = np.argsort(arf)
            
            xold = xmean.copy()
            xmean = np.zeros(n)
            zmean = np.zeros(n)
            for k in range(mu):
                xmean += weights[k] * arx[idx[k]]
                zmean += weights[k] * arz[idx[k]]
            xmean = np.clip(xmean, lower, upper)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * zmean
            hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1))) / chiN < 1.4 + 2 / (n + 1) else 0
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / max(sigma, 1e-30)
            
            artmp = (arx[idx[:mu]] - xold) / max(sigma, 1e-30)
            C = (1 - c1 - cmu_val) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for k in range(mu):
                C += cmu_val * weights[k] * np.outer(artmp[k], artmp[k])
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-15:
                C += (1e-15 - np.min(eigvals)) * np.eye(n)
            
            sigma *= np.exp((cs / ds) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, np.max(rng))
            
            if sigma < 1e-15 * np.mean(rng):
                break
    
    # === Phase 1: Global SHADE search (50% of time) ===
    global_end = 0.50
    
    while time_ok(global_end):
        generation += 1
        S_F, S_CR, S_d
