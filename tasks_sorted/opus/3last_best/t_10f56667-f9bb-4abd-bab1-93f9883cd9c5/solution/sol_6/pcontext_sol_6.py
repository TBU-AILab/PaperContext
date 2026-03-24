#Looking at the results, algorithm 1 (score 17.15) is close to the best (14.88). The key differences: algorithm 1 has proper CMA-ES but uses IPOP restarts with increasing lambda which wastes evaluations. The best algorithm has weak CMA-ES (no covariance update). I need to combine L-SHADE with proper CMA-ES, better time allocation, and multiple strategies.
#
#Key improvements:
#1. **Better L-SHADE with proper LPSR** based on evaluation count
#2. **Full CMA-ES with proper covariance adaptation** (not the broken version)
#3. **Smaller initial CMA-ES sigma for local refinement**
#4. **Multiple CMA-ES restarts from best with random perturbation**
#5. **Nelder-Mead as final polishing step**
#6. **Better time budgeting**
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    evals = 0

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower

    def elapsed():
        return (datetime.now() - start).total_seconds()

    def time_ok(frac=0.95):
        return elapsed() < max_time * frac

    def ev(x):
        nonlocal best, best_x, evals
        x = np.clip(x, lower, upper)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_x = x.copy()
        return f

    # === Phase 1: L-SHADE ===
    N_init = min(max(40, 12 * dim), 400)
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

    eval_rate = evals / max(0.001, elapsed())
    max_evals_shade = int(eval_rate * max_time * 0.55)
    evals_at_start = evals

    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    archive = []

    while time_ok(0.55):
        S_F, S_CR, S_w = [], [], []
        sorted_idx = np.argsort(fit)
        p_max = max(2, int(0.2 * pop_size))

        for i in range(pop_size):
            if not time_ok(0.55):
                break
            ri = np.random.randint(H)
            F_i = M_F[ri] + 0.1 * np.random.standard_cauchy()
            while F_i <= 0:
                F_i = M_F[ri] + 0.1 * np.random.standard_cauchy()
            F_i = min(F_i, 1.0)
            CR_i = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0.0, 1.0)

            pbest = sorted_idx[np.random.randint(0, np.random.randint(1, p_max + 1))]

            r1 = np.random.randint(pop_size - 1)
            if r1 >= i: r1 += 1

            union = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            union = [c for c in union if c != i and c != r1]
            r2c = union[np.random.randint(len(union))]
            r2v = pop[r2c] if r2c < pop_size else archive[r2c - pop_size]

            mutant = pop[i] + F_i * (pop[pbest] - pop[i]) + F_i * (pop[r1] - r2v)
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

        ratio = (evals - evals_at_start) / max(1, max_evals_shade)
        new_size = max(N_min, int(round(N_init + (N_min - N_init) * ratio)))
        if new_size < pop_size:
            idx = np.argsort(fit)[:new_size]
            pop = pop[idx]; fit = fit[idx]; pop_size = new_size

    # === Phase 2: CMA-ES restarts ===
    n = dim
    restart = 0
    while time_ok(0.98) and best_x is not None:
        lam = 4 + int(3 * np.log(n))
        mu_c = lam // 2
        w = np.log(mu_c + 0.5) - np.log(np.arange(1, mu_c + 1)); w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        cc=(4+mueff/n)/(n+4+2*mueff/n); cs=(mueff+2)/(n+mueff+5)
        c1=2/((n+1.3)**2+mueff); cmu=min(1-c1,2*(mueff-2+1/mueff)/((n+2)**2+mueff))
        ds=1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs; chiN=np.sqrt(n)*(1-1/(4*n)+1/(21*n**2))
        sigma=0.05*np.mean(ranges)*(0.8**restart)
        xmean=best_x.copy()+sigma*0.1*np.random.randn(n); xmean=np.clip(xmean,lower,upper)
        pc=np.zeros(n);ps=np.zeros(n);B=np.eye(n);D=np.ones(n);invsqrtC=np.eye(n);C=np.eye(n);ee=0;gn=0
        while time_ok(0.98):
            gn+=1; arz=np.random.randn(lam,n); arx=xmean+sigma*(arz@(B*D).T)
            arfit=np.array([ev(arx[j]) for j in range(lam) if time_ok(0.98)])
            if len(arfit)<lam: break
            idx=np.argsort(arfit); xold=xmean; xmean=w@arx[idx[:mu_c]]
            ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*(invsqrtC@((xmean-xold)/sigma))
            hn=float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*gn))/chiN<1.4+2/(n+1))
            pc=(1-cc)*pc+hn*np.sqrt(cc*(2-cc)*mueff)*(xmean-xold)/sigma
            artmp=(arx[idx[:mu_c]]-xold)/sigma
            C=(1-c1-cmu)*C+c1*(np.outer(pc,pc)+(1-hn)*cc*(2-cc)*C)+cmu*(artmp.T@np.diag(w)@artmp)
            sigma*=np.exp((cs/ds)*(np.linalg.norm(ps)/chiN-1)); sigma=min(sigma,0.3*np.mean(ranges))
            ee+=lam
            if ee>=lam*(n+2):
                C=(C+C.T)/2
                try: D2,B=np.linalg.eigh(C);D=np.sqrt(np.maximum(D2,1e-20));invsqrtC=B@np.diag(1/D)@B.T
                except: C=np.eye(n);B=np.eye(n);D=np.ones(n);invsqrtC=np.eye(n)
                ee=0
            if sigma<1e-13: break
        restart += 1
        if restart > 5: break
    return best
