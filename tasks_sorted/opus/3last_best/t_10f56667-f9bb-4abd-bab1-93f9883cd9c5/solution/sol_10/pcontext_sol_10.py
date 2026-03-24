#Looking at the results, the best algorithm (12.78) uses L-SHADE + CMA-ES restarts. I'll improve it with:
#
#1. **Better initial sampling** with opposition-based learning
#2. **Improved CMA-ES** with stagnation detection and diverse restarts from top solutions
#3. **Adaptive time allocation** - more time for CMA-ES refinement
#4. **Coordinate-wise local search** as final polish
#5. **Better restart strategy** - alternate between local refinement and broader exploration
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    evals = 0
    top_solutions = []

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
        # Maintain top solutions for diverse restarts
        if len(top_solutions) < 20:
            top_solutions.append((f, x.copy()))
            top_solutions.sort(key=lambda t: t[0])
        elif f < top_solutions[-1][0]:
            top_solutions[-1] = (f, x.copy())
            top_solutions.sort(key=lambda t: t[0])
        return f

    n = dim

    # === Phase 1: L-SHADE ===
    N_init = min(max(40, 14 * dim), 500)
    N_min = max(4, dim)
    pop_size = N_init
    H = 100

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

    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0
    archive = []
    evals_at_start = evals

    eval_rate = evals / max(0.001, elapsed())
    max_evals_shade = int(eval_rate * max_time * 0.50)

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

            r1 = np.random.randint(pop_size - 1)
            if r1 >= i: r1 += 1

            union = [c for c in range(pop_size) if c != i and c != r1]
            arc_ids = list(range(len(archive)))
            all_r2 = union + [pop_size + a for a in arc_ids]
            if not all_r2:
                continue
            r2c = all_r2[np.random.randint(len(all_r2))]
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
        new_size = max(N_min, int(round(N_init + (N_min - N_init) * min(ratio, 1.0))))
        if new_size < pop_size:
            idx = np.argsort(fit)[:new_size]
            pop = pop[idx]; fit = fit[idx]; pop_size = new_size

    # === Phase 2: CMA-ES restarts from diverse top solutions ===
    restart = 0
    while time_ok(0.92) and best_x is not None:
        lam = 4 + int(3 * np.log(n))
        mu_c = lam // 2
        ww = np.log(mu_c + 0.5) - np.log(np.arange(1, mu_c + 1)); ww /= ww.sum()
        mueff = 1.0 / np.sum(ww**2)
        cc=(4+mueff/n)/(n+4+2*mueff/n); cs=(mueff+2)/(n+mueff+5)
        c1=2/((n+1.3)**2+mueff); cmu=min(1-c1,2*(mueff-2+1/mueff)/((n+2)**2+mueff))
        ds=1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs; chiN=np.sqrt(n)*(1-1/(4*n)+1/(21*n**2))
        
        # Alternate: start from best or from a diverse top solution
        if restart == 0 or restart % 3 == 0:
            x0 = best_x.copy()
            sigma = 0.03 * np.mean(ranges)
        elif restart % 3 == 1 and len(top_solutions) > 3:
            idx_choice = min(restart, len(top_solutions) - 1)
            x0 = top_solutions[idx_choice][1].copy()
            sigma = 0.05 * np.mean(ranges)
        else:
            x0 = best_x.copy() + 0.1 * ranges * np.random.randn(n)
            sigma = 0.08 * np.mean(ranges) * (0.7 ** (restart // 3))
        
        sigma = max(sigma, 1e-8 * np.mean(ranges))
        xmean = np.clip(x0, lower, upper)
        pc=np.zeros(n);ps=np.zeros(n);B=np.eye(n);D=np.ones(n);invsqrtC=np.eye(n);C=np.eye(n);ee=0;gn=0
        stag=0;prev_best_cma=best
        
        while time_ok(0.92):
            gn+=1; arz=np.random.randn(lam,n); arx=xmean+sigma*(arz@(B*D).T)
            arfit=[]
            for j in range(lam):
                if not time_ok(0.92): break
                arfit.append(ev(arx[j]))
            if len(arfit)<lam: break
            arfit=np.array(arfit)
            idx=np.argsort(arfit); xold=xmean; xmean=ww@arx[idx[:mu_c]]
            ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*(invsqrtC@((xmean-xold)/sigma))
            hn=float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*gn))/chiN<1.4+2/(n+1))
            pc=(1-cc)*pc+hn*np.sqrt(cc*(2-cc)*mueff)*(xmean-xold)/sigma
            artmp=(arx[idx[:mu_c]]-xold)/sigma
            C=(1-c1-cmu)*C+c1*(np.outer(pc,pc)+(1-hn)*cc*(2-cc)*C)+cmu*(artmp.T@np.diag(ww)@artmp)
            sigma*=np.exp((cs/ds)*(np.linalg.norm(ps)/chiN-1)); sigma=min(sigma,0.3*np.mean(ranges))
            ee+=lam
            if ee>=lam*max(1,(n+2)//2):
                C=(C+C.T)/2
                try: D2,B=np.linalg.eigh(C);D=np.sqrt(np.maximum(D2,1e-20));invsqrtC=B@np.diag(1/D)@B.T
                except: C=np.eye(n);B=np.eye(n);D=np.ones(n);invsqrtC=np.eye(n)
                ee=0
            if sigma<1e-14: break
            if gn % 15 == 0:
                if best >= prev_best_cma - 1e-12: stag += 1
                else: stag = 0
                prev_best_cma = best
                if stag >= 3: break
        restart += 1
        if restart > 12: break

    # === Phase 3: Coordinate-wise local search ===
    if time_ok(0.97) and best_x is not None:
        x = best_x.copy()
        fx = best
        step = 0.001 * ranges
        improved = True
        while improved and time_ok(0.99):
            improved = False
            for i in range(n):
                if not time_ok(0.99): break
                for sign in [1, -1]:
                    xt = x.copy()
                    xt[i] = np.clip(x[i] + sign * step[i], lower[i], upper[i])
                    ft = ev(xt)
                    if ft < fx:
                        # Accelerate in this direction
                        x = xt; fx = ft; improved = True
                        for _ in range(5):
                            if not time_ok(0.99): break
                            xt2 = x.copy()
                            xt2[i] = np.clip(x[i] + sign * step[i] * 2, lower[i], upper[i])
                            ft2 = ev(xt2)
                            if ft2 < fx:
                                x = xt2; fx = ft2
                                step[i] *= 1.5
                            else:
                                break
                        break
            if not improved:
                step *= 0.5
                if np.max(step / ranges) > 1e-10:
                    improved = True  # try again with smaller step

    return best
