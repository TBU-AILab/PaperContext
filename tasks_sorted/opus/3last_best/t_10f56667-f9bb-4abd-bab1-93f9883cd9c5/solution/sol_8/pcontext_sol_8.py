#Looking at the results, the best algorithm (12.78) uses L-SHADE + CMA-ES restarts with decreasing sigma. Key issues I see:
#
#1. **L-SHADE time allocation could be better** - need adaptive splitting
#2. **CMA-ES restarts need stagnation detection** to avoid wasting time
#3. **Need better diversity in CMA-ES restarts** - not just from best_x but from diverse good solutions
#4. **Add coordinate-wise local search** as final polishing
#5. **Eigendecomposition frequency** should be more frequent for small dim
#6. **Try error-handling for numerical issues in CMA-ES**
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    evals = 0
    top_solutions = []  # store diverse good solutions

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
        # Track diverse good solutions
        if len(top_solutions) < 20:
            top_solutions.append((f, x.copy()))
            top_solutions.sort(key=lambda t: t[0])
        elif f < top_solutions[-1][0]:
            top_solutions[-1] = (f, x.copy())
            top_solutions.sort(key=lambda t: t[0])
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

    # Opposition-based learning
    if time_ok(0.45):
        opp = lower + upper - pop
        opp_fit = []
        for i in range(pop_size):
            if not time_ok(0.45):
                break
            opp_fit.append(ev(opp[i]))
        if len(opp_fit) == pop_size:
            opp_fit = np.array(opp_fit)
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

    gen = 0
    while time_ok(0.50):
        gen += 1
        S_F, S_CR, S_w = [], [], []
        sorted_idx = np.argsort(fit)
        p_max = max(2, int(0.2 * pop_size))

        for i in range(pop_size):
            if not time_ok(0.50):
                break
            ri = np.random.randint(H)
            mu_f = M_F[ri]
            F_i = mu_f + 0.1 * np.random.standard_cauchy()
            while F_i <= 0:
                F_i = mu_f + 0.1 * np.random.standard_cauchy()
            F_i = min(F_i, 1.0)

            mu_cr = M_CR[ri]
            if mu_cr < 0:
                CR_i = 0.0
            else:
                CR_i = np.clip(mu_cr + 0.1 * np.random.randn(), 0.0, 1.0)

            pi = max(2, int(p_max * np.random.random()) + 1)
            pbest = sorted_idx[np.random.randint(0, min(pi, pop_size))]

            candidates = list(range(pop_size))
            candidates.remove(i)
            r1 = candidates[np.random.randint(len(candidates))]

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
            ww = np.array(S_w); ww /= ww.sum() + 1e-30
            sf = np.array(S_F)
            M_F[k] = np.sum(ww * sf**2) / (np.sum(ww * sf) + 1e-30)
            scr = np.array(S_CR)
            if np.max(scr) == 0:
                M_CR[k] = -1.0
            else:
                M_CR[k] = np.sum(ww * scr**2) / (np.sum(ww * scr) + 1e-30)
            k = (k + 1) % H

        ratio = (evals - evals_at_start) / max(1, max_evals_shade)
        new_size = max(N_min, int(round(N_init + (N_min - N_init) * min(ratio, 1.0))))
        if new_size < pop_size:
            idx = np.argsort(fit)[:new_size]
            pop = pop[idx]; fit = fit[idx]; pop_size = new_size

    # === Phase 2: CMA-ES restarts from diverse good solutions ===
    restart = 0
    while time_ok(0.93) and best_x is not None:
        lam = 4 + int(3 * np.log(n))
        mu_c = lam // 2
        w = np.log(mu_c + 0.5) - np.log(np.arange(1, mu_c + 1)); w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        cc=(4+mueff/n)/(n+4+2*mueff/n); cs=(mueff+2)/(n+mueff+5)
        c1=2/((n+1.3)**2+mueff); cmu=min(1-c1,2*(mueff-2+1/mueff)/((n+2)**2+mueff))
        ds=1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs; chiN=np.sqrt(n)*(1-1/(4*n)+1/(21*n**2))
        sigma=0.05*np.mean(ranges)*(0.7**restart)
        if sigma < 1e-12: sigma = 0.01 * np.mean(ranges)
        
        # Pick start point: alternate between best and diverse solutions
        if restart == 0 or restart % 3 == 0:
            xmean = best_x.copy()
        elif len(top_solutions) > min(restart, len(top_solutions)-1):
            idx_s = min(restart, len(top_solutions)-1)
            xmean = top_solutions[idx_s][1].copy()
        else:
            xmean = best_x.copy()
        xmean += sigma * 0.1 * np.random.randn(n)
        xmean = np.clip(xmean, lower, upper)
        
        pc=np.zeros(n);ps=np.zeros(n);B=np.eye(n);D=np.ones(n);invsqrtC=np.eye(n);C=np.eye(n);eigeneval=0;gn=0
        best_in_run = float('inf'); stag_count = 0
        
        while time_ok(0.93):
            gn += 1
            arz = np.random.randn(lam, n)
            arx = xmean + sigma * (arz @ (B * D).T)
            arfit = []
            for j in range(lam):
                if not time_ok(0.93): break
                arfit.append(ev(arx[j]))
            if len(arfit) < lam: break
            arfit = np.array(arfit)
            
            idx = np.argsort(arfit)
            xold = xmean
            xmean = w @ arx[idx[:mu_c]]
            
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * (invsqrtC @ ((xmean-xold)/sigma))
            hn = float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*gn))/chiN < 1.4+2/(n+1))
            pc = (1-cc)*pc + hn*np.sqrt(cc*(2-cc)*mueff) * (xmean-xold)/sigma
            artmp = (arx[idx[:mu_c]] - xold) / sigma
            C = (1-c1-cmu)*C + c1*(np.outer(pc,pc) + (1-hn)*cc*(2-cc)*C) + cmu*(artmp.T @ np.diag(w) @ artmp)
            sigma *= np.exp((cs/ds)*(np.linalg.norm(ps)/chiN - 1))
            sigma = min(sigma, 0.3*np.mean(ranges))
            
            eigeneval += lam
            decomp_freq = max(1, lam * n // 10)
            if eigeneval >= decomp_freq:
                C = (C + C.T) / 2
                try:
                    D2, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D2, 1e-20))
                    invsqrtC = B @ np.diag(1/D) @ B.T
                except:
                    C = np.eye(n); B = np.eye(n); D = np.ones(n); invsqrtC = np.eye(n)
                eigeneval = 0
            
            cur_best = np.min(arfit)
            if cur_best < best_in_run - 1e-10:
                best_in_run = cur_best
                stag_count = 0
            else:
                stag_count += 1
            
            if sigma < 1e-14 or stag_count > 10 + 3*n: break
            if np.max(D) / np.min(D) > 1e7: break
            
        restart += 1
        if restart > 10: break

    # === Phase 3: Coordinate-wise local search + Nelder-Mead polishing ===
    if time_ok(0.97) and best_x is not None:
        # Coordinate descent
        x = best_x.copy()
        fx = best
        step = 0.001 * ranges
        improved = True
        while improved and time_ok(0.98):
            improved = False
            for d in range(n):
                if not time_ok(0.98): break
                for sign in [1, -1]:
                    xn = x.copy()
                    xn[d] += sign * step[d]
                    xn = np.clip(xn, lower, upper)
                    fn = ev(xn)
                    if fn < fx:
                        # Keep going in this direction
                        while time_ok(0.98):
                            step[d] *= 1.5
                            xn2 = xn.copy()
                            xn2[d] += sign * step[d]
                            xn2 = np.clip(xn2, lower, upper)
                            fn2 = ev(xn2)
                            if fn2 < fn:
                                xn = xn2; fn = fn2
                            else:
                                break
                        x = xn; fx = fn; improved = True
                        step[d] = max(step[d] * 0.5, 1e-10 * ranges[d])
                        break
                else:
                    step[d] *= 0.5

    # Nelder-Mead polishing
    if time_ok(0.99) and best_x is not None:
        step_nm = 0.0005 * ranges
        simplex = [best_x.copy()]
        for i in range(n):
            p = best_x.copy(); p[i] += step_nm[i]; simplex.append(np.clip(p, lower, upper))
        simplex = np.array(simplex)
        sf = np.array([ev(simplex[j]) for j in range(n+1) if time_ok(0.995)])
        if len(sf) == n+1:
            while time_ok(0.998):
                order = np.argsort(sf); simplex = simplex[order]; sf = sf[order]
                centroid = simplex[:-1].mean(axis=0)
                xr = 2*centroid - simplex[-1]; fr = ev(np.clip(xr, lower, upper))
                if fr < sf[0]:
                    xe = 3*centroid - 2*simplex[-1]; fe = ev(np.clip(xe, lower, upper))
                    if fe < fr: simplex[-1]=xe; sf[-1]=fe
                    else: simplex[-1]=xr; sf[-1]=fr
                elif fr < sf[-2]: simplex[-1]=xr; sf[-1]=fr
                else:
                    if fr < sf[-1]: simplex[-1]=xr; sf[-1]=fr
                    xc = 0.5*(centroid+simplex[-1]); fc = ev(np.clip(xc,
