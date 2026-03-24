#Looking at the code, I can identify several improvements:
#
#1. **The CMA-ES local search is weak** - it doesn't properly update covariance, just decays sigma
#2. **L-SHADE with linear population size reduction (LPSR)** would be better
#3. **Need proper CMA-ES with evolution paths and covariance adaptation**
#4. **Per-individual Python loops are slow** - need to vectorize where possible
#5. **Restart strategy for CMA-ES** when stuck
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def time_ok(frac=0.95):
        return elapsed() < max_time * frac
    
    def ev(x):
        nonlocal best, best_x
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # === Phase 1: L-SHADE ===
    N_init = min(max(40, 10 * dim), 400)
    N_min = max(4, dim)
    pop_size = N_init
    H = 100
    
    pop = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, i] = (perm + pop[:, i]) / pop_size
    pop = lower + pop * ranges
    
    fit = np.array([ev(pop[i]) for i in range(pop_size)])
    if not time_ok(0.5):
        return best
    
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    archive = []
    gen = 0
    total_evals_est = int(max_time * 0.55 * pop_size / max(0.001, elapsed() / pop_size))
    max_gen_est = max(1, total_evals_est // pop_size)

    while time_ok(0.55):
        gen += 1
        S_F, S_CR, S_w = [], [], []
        sorted_idx = np.argsort(fit)
        p_max = max(2, int(0.15 * pop_size))

        # Generate all F and CR values at once
        ri_all = np.random.randint(0, H, pop_size)
        F_all = M_F[ri_all] + 0.1 * np.random.standard_cauchy(pop_size)
        CR_all = np.clip(M_CR[ri_all] + 0.1 * np.random.randn(pop_size), 0.0, 1.0)
        
        for i in range(pop_size):
            if not time_ok(0.55):
                break
            
            F_i = F_all[i]
            while F_i <= 0:
                F_i = M_F[ri_all[i]] + 0.1 * np.random.standard_cauchy()
            F_i = min(F_i, 1.0)
            CR_i = CR_all[i]
            
            pbest = sorted_idx[np.random.randint(0, np.random.randint(1, p_max + 1))]
            
            r1 = np.random.randint(pop_size - 1)
            if r1 >= i: r1 += 1
            
            union_size = pop_size + len(archive)
            r2 = np.random.randint(union_size - 1)
            candidates = list(range(pop_size)) + list(range(pop_size, union_size))
            candidates = [c for c in candidates if c != i and c != r1]
            r2c = candidates[np.random.randint(len(candidates))]
            r2_vec = pop[r2c] if r2c < pop_size else archive[r2c - pop_size]
            
            mutant = pop[i] + F_i * (pop[pbest] - pop[i]) + F_i * (pop[r1] - r2_vec)
            
            mask = np.random.random(dim) < CR_i
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, pop[i])
            
            lo = trial < lower; hi = trial > upper
            trial[lo] = (lower[lo] + pop[i][lo]) / 2
            trial[hi] = (upper[hi] + pop[i][hi]) / 2
            
            f_trial = ev(trial)
            if f_trial <= fit[i]:
                d = fit[i] - f_trial
                if d > 0:
                    S_F.append(F_i); S_CR.append(CR_i); S_w.append(d)
                archive.append(pop[i].copy())
                if len(archive) > N_init:
                    archive.pop(np.random.randint(len(archive)))
                pop[i] = trial; fit[i] = f_trial
        
        if S_F:
            w = np.array(S_w); w /= w.sum() + 1e-30
            sf = np.array(S_F)
            M_F[k] = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * np.array(S_CR))
            k = (k + 1) % H
        
        new_size = max(N_min, int(round(N_init - (N_init - N_min) * gen / max(1, max_gen_est))))
        if new_size < pop_size:
            idx = np.argsort(fit)[:new_size]
            pop = pop[idx]; fit = fit[idx]; pop_size = new_size

    # === Phase 2: CMA-ES with restarts ===
    while time_ok(0.98) and best_x is not None:
        n = dim
        sigma = 0.1 * np.mean(ranges)
        xmean = best_x.copy() + 0.001 * ranges * np.random.randn(n)
        xmean = np.clip(xmean, lower, upper)
        lam = 4 + int(3 * np.log(n))
        mu_c = lam // 2
        w = np.log(mu_c + 0.5) - np.log(np.arange(1, mu_c + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        cc = (4+mueff/n)/(n+4+2*mueff/n)
        cs = (mueff+2)/(n+mueff+5)
        c1 = 2/((n+1.3)**2+mueff)
        cmu = min(1-c1, 2*(mueff-2+1/mueff)/((n+2)**2+mueff))
        ds = 1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs
        chiN = np.sqrt(n)*(1-1/(4*n)+1/(21*n**2))
        pc=np.zeros(n); ps=np.zeros(n)
        B=np.eye(n); D=np.ones(n); invsqrtC=np.eye(n)
        eigeneval=0; counteval=0
        
        while time_ok(0.98):
            arz = np.random.randn(lam, n)
            arx = xmean + sigma*(arz@(B*D).T)
            arfit = np.array([ev(arx[j]) for j in range(lam) if time_ok(0.98)])
            if len(arfit)<lam: break
            counteval += lam
            idx = np.argsort(arfit)
            xold=xmean.copy()
            xmean = w@arx[idx[:mu_c]]
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff)*(invsqrtC@((xmean-xold)/sigma))
            hsig = float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*counteval/lam))/chiN < 1.4+2/(n+1))
            pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mueff)*(xmean-xold)/sigma
            artmp = (arx[idx[:mu_c]]-xold)/sigma
            C = (1-c1-cmu)*np.outer(B*D,B*D).reshape(n,n)*0+(1-c1-cmu)*(B@np.diag(D**2)@B.T)+c1*(np.outer(pc,pc)+(1-hsig)*cc*(2-cc)*(B@np.diag(D**2)@B.T))+cmu*(artmp.T@np.diag(w)@artmp)
            sigma *= np.exp((cs/ds)*(np.linalg.norm(ps)/chiN-1))
            sigma = min(sigma, 0.5*np.mean(ranges))
            eigeneval += lam
            if eigeneval >= lam*(n+2):
                C=(C+C.T)/2
                D2,B=np.linalg.eigh(C)
                D=np.sqrt(np.maximum(D2,1e-20))
                invsqrtC=B@np.diag(1/D)@B.T
                eigeneval=0
            if sigma<1e-12: break
    
    return best
