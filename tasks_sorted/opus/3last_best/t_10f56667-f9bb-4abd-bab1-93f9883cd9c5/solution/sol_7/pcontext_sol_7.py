#Looking at the results, the best algorithm (12.78) uses L-SHADE + CMA-ES restarts with decreasing sigma. Key issues to address:
#
#1. **L-SHADE time allocation** - 55% might be too much or too little depending on the problem
#2. **CMA-ES restarts** - limited to 5 restarts, could benefit from more diversity
#3. **Need Nelder-Mead polishing** for final refinement
#4. **Better initial sampling** - add opposition-based learning
#5. **Adaptive time splitting** based on improvement rate
#6. **Add a diversified search phase** between L-SHADE and CMA-ES
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

    n = dim

    # === Phase 1: L-SHADE ===
    N_init = min(max(50, 14 * dim), 500)
    N_min = max(4, dim)
    pop_size = N_init
    H = 100

    # Latin hypercube sampling
    pop = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        pop[:, i] = (perm + pop[:, i]) / pop_size
    pop = lower + pop * ranges

    fit = np.array([ev(pop[i]) for i in range(pop_size)])
    
    # Opposition-based learning for initial pop
    if time_ok(0.5):
        opp = lower + upper - pop
        opp_fit = np.array([ev(opp[i]) for i in range(pop_size) if time_ok(0.5)])
        if len(opp_fit) == pop_size:
            combined = np.vstack([pop, opp])
            combined_fit = np.concatenate([fit, opp_fit])
            idx = np.argsort(combined_fit)[:pop_size]
            pop = combined[idx]
            fit = combined_fit[idx]

    if not time_ok(0.4):
        return best

    eval_rate = evals / max(0.001, elapsed())
    max_evals_shade = int(eval_rate * max_time * 0.50)
    evals_at_start = evals

    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    archive = []

    while time_ok(0.50):
        S_F, S_CR, S_w = [], [], []
        sorted_idx = np.argsort(fit)
        p_max = max(2, int(0.2 * pop_size))

        for i in range(pop_size):
            if not time_ok(0.50):
                break
            ri = np.random.randint(H)
            F_i = M_F[ri] + 0.1 * np.random.standard_cauchy()
            while F_i <= 0:
                F_i = M_F[ri] + 0.1 * np.random.standard_cauchy()
            F_i = min(F_i, 1.0)
            CR_i = np.clip(M_CR[ri] + 0.1 * np.random.randn(), 0.0, 1.0)

            pbest = sorted_idx[np.random.randint(0, np.random.randint(1, p_max + 1))]

            candidates = list(range(pop_size))
            candidates.remove(i)
            r1 = candidates[np.random.randint(len(candidates))]

            union_pop = list(range(pop_size))
            union_arc = list(range(len(archive)))
            all_exc = [('p', c) for c in range(pop_size) if c != i and c != r1] + [('a', c) for c in range(len(archive))]
            if not all_exc:
                continue
            tag, r2c = all_exc[np.random.randint(len(all_exc))]
            r2v = pop[r2c] if tag == 'p' else archive[r2c]

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
        new_size = max(N_min, int(round(N_init + (N_min - N_init) * min(ratio, 1.0))))
        if new_size < pop_size:
            idx = np.argsort(fit)[:new_size]
            pop = pop[idx]; fit = fit[idx]; pop_size = new_size

    # === Phase 2: CMA-ES restarts with decreasing sigma ===
    restart = 0
    while time_ok(0.95) and best_x is not None:
        lam = 4 + int(3 * np.log(n))
        mu_c = lam // 2
        ww = np.log(mu_c + 0.5) - np.log(np.arange(1, mu_c + 1)); ww /= ww.sum()
        mueff = 1.0 / np.sum(ww**2)
        cc=(4+mueff/n)/(n+4+2*mueff/n); cs=(mueff+2)/(n+mueff+5)
        c1=2/((n+1.3)**2+mueff); cmu=min(1-c1,2*(mueff-2+1/mueff)/((n+2)**2+mueff))
        ds=1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs; chiN=np.sqrt(n)*(1-1/(4*n)+1/(21*n**2))
        sigma=0.05*np.mean(ranges)*(0.7**restart)
        if sigma < 1e-10: break
        xmean=best_x.copy()+sigma*0.1*np.random.randn(n); xmean=np.clip(xmean,lower,upper)
        pc=np.zeros(n);ps=np.zeros(n);B=np.eye(n);D=np.ones(n);invsqrtC=np.eye(n);C=np.eye(n);ee=0;gn=0
        stag = 0; prev_best = best
        while time_ok(0.95):
            gn+=1; arz=np.random.randn(lam,n); arx=xmean+sigma*(arz@(B*D).T)
            arfit=np.array([ev(arx[j]) for j in range(lam) if time_ok(0.95)])
            if len(arfit)<lam: break
            idx=np.argsort(arfit); xold=xmean; xmean=ww@arx[idx[:mu_c]]
            ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*(invsqrtC@((xmean-xold)/sigma))
            hn=float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*gn))/chiN<1.4+2/(n+1))
            pc=(1-cc)*pc+hn*np.sqrt(cc*(2-cc)*mueff)*(xmean-xold)/sigma
            artmp=(arx[idx[:mu_c]]-xold)/sigma
            C=(1-c1-cmu)*C+c1*(np.outer(pc,pc)+(1-hn)*cc*(2-cc)*C)+cmu*(artmp.T@np.diag(ww)@artmp)
            sigma*=np.exp((cs/ds)*(np.linalg.norm(ps)/chiN-1)); sigma=min(sigma,0.3*np.mean(ranges))
            ee+=lam
            if ee>=lam*(n+2):
                C=(C+C.T)/2
                try: D2,B=np.linalg.eigh(C);D=np.sqrt(np.maximum(D2,1e-20));invsqrtC=B@np.diag(1/D)@B.T
                except: C=np.eye(n);B=np.eye(n);D=np.ones(n);invsqrtC=np.eye(n)
                ee=0
            if sigma<1e-14: break
            if gn % 20 == 0:
                if best >= prev_best - 1e-10: stag += 1
                else: stag = 0
                prev_best = best
                if stag >= 3: break
        restart += 1
        if restart > 8: break

    # === Phase 3: Nelder-Mead polishing ===
    if time_ok(0.99) and best_x is not None:
        step = 0.001 * ranges
        simplex = [best_x.copy()]
        for i in range(n):
            p = best_x.copy(); p[i] += step[i]; simplex.append(p)
        simplex = np.array(simplex)
        sf = np.array([ev(simplex[j]) for j in range(n+1) if time_ok(0.99)])
        if len(sf) == n+1:
            while time_ok(0.995):
                order = np.argsort(sf); simplex = simplex[order]; sf = sf[order]
                centroid = simplex[:-1].mean(axis=0)
                xr = 2*centroid - simplex[-1]; fr = ev(xr)
                if fr < sf[0]:
                    xe = 3*centroid - 2*simplex[-1]; fe = ev(xe)
                    if fe < fr: simplex[-1]=xe; sf[-1]=fe
                    else: simplex[-1]=xr; sf[-1]=fr
                elif fr < sf[-2]: simplex[-1]=xr; sf[-1]=fr
                else:
                    if fr < sf[-1]: simplex[-1]=xr; sf[-1]=fr
                    xc = 0.5*(centroid+simplex[-1]); fc = ev(xc)
                    if fc <= sf[-1]: simplex[-1]=xc; sf[-1]=fc
                    else:
                        for j in range(1,n+1):
                            simplex[j]=0.5*(simplex[0]+simplex[j])
                            sf[j]=ev(simplex[j])
                            if not time_ok(0.995): break

    return best
