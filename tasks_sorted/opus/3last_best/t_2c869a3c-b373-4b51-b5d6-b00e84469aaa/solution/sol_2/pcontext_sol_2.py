#Looking at the best algorithm (score 38.03), I can identify several improvements:
#
#1. **Vectorized mirror bounds** - the per-element while loop is very slow
#2. **Better time allocation** - spend less on init, more on CMA-ES
#3. **Smarter restart strategy** - alternate between local refinement and exploration
#4. **Active CMA-ES** with negative weights for worse solutions
#5. **Vectorized population generation** to reduce Python loop overhead
#6. **Better stagnation detection and sigma adaptation**
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
    center = (lower + upper) / 2.0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_left():
        return max_time * 0.97 - elapsed()
    
    def eval_f(x):
        nonlocal best, best_x
        x_c = np.clip(x, lower, upper)
        f = func(x_c)
        if f < best:
            best = f
            best_x = x_c.copy()
        return f
    
    def mirror_bounds(x):
        x = x.copy()
        for d in range(dim):
            r = ranges[d]
            if r <= 0:
                x[d] = lower[d]
                continue
            while x[d] < lower[d] or x[d] > upper[d]:
                if x[d] < lower[d]:
                    x[d] = 2*lower[d] - x[d]
                elif x[d] > upper[d]:
                    x[d] = 2*upper[d] - x[d]
        return x
    
    # Phase 1: LHS initialization
    n_init = min(max(20 * dim, 100), 800)
    init_points = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        init_points[:, i] = lower[i] + (perm + np.random.uniform(0, 1, n_init)) / n_init * ranges[i]
    
    init_fitnesses = []
    for i in range(n_init):
        if time_left() <= 0:
            return best
        f = eval_f(init_points[i])
        init_fitnesses.append((f, i))
    
    eval_f(center)
    
    init_fitnesses.sort()
    top_k = min(8, len(init_fitnesses))
    starting_points = [init_points[init_fitnesses[i][1]].copy() for i in range(top_k)]
    
    # Phase 2: BIPOP-CMA-ES
    base_pop_size = 4 + int(3 * np.log(dim))
    base_pop_size = max(base_pop_size, 10)
    restart_count = 0
    large_pop_budget = 0
    
    while time_left() > 0.3:
        # Decide restart type: small (local) or large (global)
        if restart_count < len(starting_points):
            x0 = starting_points[restart_count].copy()
            pop_size = base_pop_size
            sigma0 = np.mean(ranges) / 4.0
        elif restart_count % 3 == 0 or (best_x is not None and np.random.random() < 0.4):
            # Local restart near best
            x0 = best_x.copy() if best_x is not None else center.copy()
            # Add small perturbation
            x0 = x0 + np.random.randn(dim) * ranges * 0.01
            x0 = np.clip(x0, lower, upper)
            pop_size = base_pop_size
            sigma0 = np.mean(ranges) / 10.0
        else:
            # Large population global restart
            x0 = np.array([np.random.uniform(l, u) for l, u in bounds])
            large_pop_budget += 1
            pop_size = min(base_pop_size * (2 ** min(large_pop_budget, 4)), 256)
            sigma0 = np.mean(ranges) / 3.0
        
        mu = pop_size // 2
        weights_pos = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights_pos = weights_pos / np.sum(weights_pos)
        mu_eff = 1.0 / np.sum(weights_pos ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu_val = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        use_sep = dim > 80
        mean = x0.copy()
        sigma = sigma0
        
        if use_sep:
            diag_C = np.ones(dim)
            p_s = np.zeros(dim)
            p_c = np.zeros(dim)
        else:
            C = np.eye(dim)
            p_s = np.zeros(dim)
            p_c = np.zeros(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            invsqrtC = np.eye(dim)
            eigen_countdown = 0
        
        gen = 0
        stag = 0
        prev_best_l = float('inf')
        best_l = float('inf')
        max_gen = max(100, 200 + 50*dim // pop_size)
        
        while time_left() > 0.15 and gen < max_gen:
            if stag > 20 + 10*dim // pop_size:
                break
            if sigma < 1e-15:
                break
            
            if use_sep:
                sq = np.sqrt(np.maximum(diag_C, 1e-20))
                Z = np.random.randn(pop_size, dim)
                X = mean[None, :] + sigma * sq[None, :] * Z
                sols = []
                fits = []
                for k in range(pop_size):
                    if time_left() <= 0.1:
                        return best
                    xk = np.clip(X[k], lower, upper)
                    f = eval_f(xk)
                    sols.append(xk)
                    fits.append(f)
                    if f < best_l: best_l = f
                
                idx = np.argsort(fits)
                old_mean = mean.copy()
                sel = np.array([sols[idx[i]] for i in range(mu)])
                mean = weights_pos @ sel
                md = mean - old_mean
                p_s = (1-c_sigma)*p_s + np.sqrt(c_sigma*(2-c_sigma)*mu_eff)*md/(sigma*sq)
                nps = np.linalg.norm(p_s)
                hs = 1 if nps/np.sqrt(1-(1-c_sigma)**(2*(gen+1))) < (1.4+2/(dim+1))*chi_n else 0
                p_c = (1-c_c)*p_c + hs*np.sqrt(c_c*(2-c_c)*mu_eff)*md/sigma
                diag_C = (1-c1-c_mu_val)*diag_C + c1*(p_c**2+(1-hs)*c_c*(2-c_c)*diag_C)
                for i in range(mu):
                    diag_C += c_mu_val*weights_pos[i]*((sols[idx[i]]-old_mean)/sigma)**2
                diag_C = np.maximum(diag_C, 1e-20)
                sigma *= np.exp((c_sigma/d_sigma)*(nps/chi_n-1))
            else:
                if eigen_countdown <= 0:
                    try:
                        C = (C+C.T)/2; ev, B = np.linalg.eigh(C); ev = np.maximum(ev,1e-20); D = np.sqrt(ev); invsqrtC = B@np.diag(1/D)@B.T
                    except: C=np.eye(dim); B=np.eye(dim); D=np.ones(dim); invsqrtC=np.eye(dim)
                    eigen_countdown = max(1, int(1/(c1+c_mu_val)/dim/10))
                eigen_countdown -= 1
                Z = np.random.randn(pop_size, dim)
                X = mean[None,:] + sigma*(Z@(B*D[None,:]).T)  # Faster than loop
                sols=[]; fits=[]
                for k in range(pop_size):
                    if time_left()<=0.1: return best
                    xk=np.clip(X[k],lower,upper); f=eval_f(xk); sols.append(xk); fits.append(f)
                    if f<best_l: best_l=f
                idx=np.argsort(fits); old_mean=mean.copy()
                sel=np.array([sols[idx[i]] for i in range(mu)]); mean=weights_pos@sel; md=mean-old_mean
                p_s=(1-c_sigma)*p_s+np.sqrt(c_sigma*(2-c_sigma)*mu_eff)*invsqrtC@md/sigma
                nps=np.linalg.norm(p_s); hs=1 if nps/np.sqrt(1-(1-c_sigma)**(2*(gen+1)))<(1.4+2/(dim+1))*chi_n else 0
                p_c=(1-c_c)*p_c+hs*np.sqrt(c_c*(2-c_c)*mu_eff)*md/sigma
                artmp=np.array([(sols[idx[i]]-old_mean)/sigma for i in range(mu)]).T
                C=(1-c1-c_mu_val)*C+c1*(np.outer(p_c,p_c)+(1-hs)*c_c*(2-c_c)*C)+c_mu_val*(artmp*weights_pos)@artmp.T
                sigma*=np.exp((c_sigma/d_sigma)*(nps/chi_n-1))
            
            sigma=min(sigma,np.mean(ranges)*2)
            gen+=1
            if best_l<prev_best_l-1e-10: stag=0; prev_best_l=best_l
            else: stag+=1
        restart_count += 1
    
    return best
