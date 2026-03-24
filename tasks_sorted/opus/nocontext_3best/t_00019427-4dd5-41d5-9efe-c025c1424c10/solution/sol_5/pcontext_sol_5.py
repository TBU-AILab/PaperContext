#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def eval_func(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # Phase 1: Latin Hypercube Sampling for initial exploration
    n_init = min(100, max(20, dim * 10))
    
    init_points = []
    for i in range(n_init):
        if elapsed() >= max_time * 0.9:
            return best
        x = np.array([np.random.uniform(lower[d], upper[d]) for d in range(dim)])
        f = eval_func(x)
        init_points.append((f, x.copy()))
    
    init_points.sort(key=lambda p: p[0])
    
    # Phase 2: CMA-ES inspired search from best points
    def run_nelder_mead(x0, budget_time):
        nonlocal best, best_x
        n = len(x0)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initialize simplex
        simplex = [x0.copy()]
        scale = (upper - lower) * 0.05
        for i in range(n):
            p = x0.copy()
            p[i] = p[i] + scale[i]
            if p[i] > upper[i]:
                p[i] = x0[i] - scale[i]
            simplex.append(clip(p))
        
        f_values = []
        for p in simplex:
            if elapsed() >= budget_time:
                return
            f_values.append(eval_func(p))
        
        max_iter = 10000
        for iteration in range(max_iter):
            if elapsed() >= budget_time:
                return
            
            # Sort
            order = np.argsort(f_values)
            simplex = [simplex[i] for i in order]
            f_values = [f_values[i] for i in order]
            
            # Centroid (excluding worst)
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = eval_func(xr)
            
            if fr < f_values[0]:
                # Expansion
                xe = clip(centroid + gamma * (xr - centroid))
                fe = eval_func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    f_values[-1] = fe
                else:
                    simplex[-1] = xr
                    f_values[-1] = fr
            elif fr < f_values[-2]:
                simplex[-1] = xr
                f_values[-1] = fr
            else:
                # Contraction
                if fr < f_values[-1]:
                    xc = clip(centroid + rho * (xr - centroid))
                    fc = eval_func(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        f_values[-1] = fc
                    else:
                        # Shrink
                        for j in range(1, len(simplex)):
                            simplex[j] = clip(simplex[0] + sigma * (simplex[j] - simplex[0]))
                            f_values[j] = eval_func(simplex[j])
                            if elapsed() >= budget_time:
                                return
                else:
                    xc = clip(centroid + rho * (simplex[-1] - centroid))
                    fc = eval_func(xc)
                    if fc < f_values[-1]:
                        simplex[-1] = xc
                        f_values[-1] = fc
                    else:
                        # Shrink
                        for j in range(1, len(simplex)):
                            simplex[j] = clip(simplex[0] + sigma * (simplex[j] - simplex[0]))
                            f_values[j] = eval_func(simplex[j])
                            if elapsed() >= budget_time:
                                return
            
            # Check convergence
            if np.std(f_values) < 1e-12:
                return
    
    # Run Nelder-Mead from top candidates
    n_restarts = min(5, len(init_points))
    time_per_restart = (max_time * 0.85 - elapsed()) / max(n_restarts, 1)
    
    for i in range(n_restarts):
        if elapsed() >= max_time * 0.9:
            break
        budget = elapsed() + time_per_restart
        budget = min(budget, max_time * 0.95)
        
        # Add some perturbation for restarts > 0
        x0 = init_points[i][1].copy()
        if i > 0:
            perturb = np.random.normal(0, 0.01, dim) * (upper - lower)
            x0 = clip(x0 + perturb)
        
        run_nelder_mead(x0, budget)
    
    # Phase 3: Local random search around best
    if best_x is not None:
        scale = 0.1
        while elapsed() < max_time * 0.98:
            perturb = np.random.normal(0, scale, dim) * (upper - lower)
            candidate = clip(best_x + perturb)
            f = eval_func(candidate)
            scale *= 0.999
            if scale < 1e-8:
                scale = 0.01
    
    return best
