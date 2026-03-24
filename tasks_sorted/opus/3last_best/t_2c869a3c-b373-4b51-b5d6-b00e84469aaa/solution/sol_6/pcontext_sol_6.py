#Looking at the algorithms, the best one (38.03) uses IPOP-CMA-ES with LHS initialization and Nelder-Mead refinement. I'll improve it with:
#
#1. **Faster mirror bounds** using modular arithmetic instead of while loops
#2. **Active CMA-ES** with negative weights
#3. **Better restart strategy** - BIPOP with both small/large populations
#4. **Reduced sep-CMA threshold** (dim>50)
#5. **Better time allocation** and initial sampling
#6. **Improved local search** at the end with pattern search
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
    center = (lower + upper) / 2.0

    def elapsed():
        return (datetime.now() - start).total_seconds()
    def time_left():
        return max_time * 0.96 - elapsed()

    def eval_f(x):
        nonlocal best, best_x
        x_c = np.clip(x, lower, upper)
        f = func(x_c)
        if f < best:
            best = f
            best_x = x_c.copy()
        return f

    def mirror(x):
        x = x.copy()
        for d in range(dim):
            r = ranges[d]
            if r <= 0: x[d] = lower[d]; continue
            x[d] -= lower[d]
            p = 2*r
            x[d] = x[d] % p
            if x[d] > r: x[d] = p - x[d]
            x[d] += lower[d]
        return np.clip(x, lower, upper)

    # Phase 1: LHS init
    n_init = min(max(20*dim, 150), 800)
    pts = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        pts[:, i] = lower[i] + (perm + np.random.uniform(0,1,n_init))/n_init*ranges[i]
    
    ifits = []
    for i in range(n_init):
        if time_left() <= 0: return best
        ifits.append((eval_f(pts[i]), i))
    eval_f(center)
    ifits.sort()
    top_k = min(10, len(ifits))
    starts = [pts[ifits[i][1]].copy() for i in range(top_k)]

    # Phase 2: BIPOP-CMA-ES
    base_pop = max(4 + int(3*np.log(dim)), 10)
    rc = 0; large_n = 0

    while time_left() > 0.4:
        if rc < len(starts):
            x0 = starts[rc].copy(); ps = base_pop; sig = np.mean(ranges)/4
        elif rc % 2 == 0 and best_x is not None:
            x0 = best_x + np.random.randn(dim)*ranges*0.015
            x0 = np.clip(x0, lower, upper); ps = base_pop; sig = np.mean(ranges)/8
        else:
            x0 = np.array([np.random.uniform(l,u) for l,u in bounds])
            large_n += 1; ps = min(base_pop*(2**min(large_n,4)),256); sig = np.mean(ranges)/3

        mu = ps//2
        w = np.log(mu+.5)-np.log(np.arange(1,mu+1)); w /= w.sum()
        mue = 1/np.sum(w**2)
        cs = (mue+2)/(dim+mue+5); ds = 1+2*max(0,np.sqrt((mue-1)/(dim+1))-1)+cs
        cc = (4+mue/dim)/(dim+4+2*mue/dim); c1 = 2/((dim+1.3)**2+mue)
        cmu = min(1-c1, 2*(mue-2+1/mue)/((dim+2)**2+mue))
        chi = np.sqrt(dim)*(1-1/(4*dim)+1/(21*dim**2))
        use_sep = dim > 50
        mn = x0.copy(); sg = sig
        dC = np.ones(dim) if use_sep else None
        C = None if use_sep else np.eye(dim)
        ps_v = np.zeros(dim); pc = np.zeros(dim)
        B = np.eye(dim); D = np.ones(dim); iC = np.eye(dim); ec = 0
        g = 0; stag = 0; pbl = float('inf'); bl = float('inf')
        while time_left() > 0.15 and g < max(100,300+60*dim//ps) and stag < 25+10*dim//ps and sg > 1e-15:
            if use_sep:
                sq = np.sqrt(np.maximum(dC,1e-20)); Z = np.random.randn(ps,dim); X = mn+sg*sq*Z
            else:
                if ec<=0:
                    try: C=(C+C.T)/2; ev,B=np.linalg.eigh(C); ev=np.maximum(ev,1e-20); D=np.sqrt(ev); iC=B@np.diag(1/D)@B.T
                    except: C=np.eye(dim);B=np.eye(dim);D=np.ones(dim);iC=np.eye(dim)
                    ec=max(1,int(1/(c1+cmu)/dim/10))
                ec-=1; Z=np.random.randn(ps,dim); X=mn+sg*(Z@(B*D).T)
            sl=[]; fl=[]
            for k in range(ps):
                if time_left()<=.1: return best
                xk=mirror(X[k]); f=eval_f(xk); sl.append(xk); fl.append(f)
                if f<bl: bl=f
            ix=np.argsort(fl); om=mn.copy(); sel=np.array([sl[ix[i]] for i in range(mu)]); mn=w@sel; md=mn-om
            if use_sep:
                ps_v=(1-cs)*ps_v+np.sqrt(cs*(2-cs)*mue)*md/(sg*sq)
            else:
                ps_v=(1-cs)*ps_v+np.sqrt(cs*(2-cs)*mue)*iC@md/sg
            nps=np.linalg.norm(ps_v); hs=1 if nps/np.sqrt(1-(1-cs)**(2*(g+1)))<(1.4+2/(dim+1))*chi else 0
            pc=(1-cc)*pc+hs*np.sqrt(cc*(2-cc)*mue)*md/sg
            if use_sep:
                dC=(1-c1-cmu)*dC+c1*(pc**2+(1-hs)*cc*(2-cc)*dC)
                for i in range(mu): dC+=cmu*w[i]*((sl[ix[i]]-om)/sg)**2
                dC=np.maximum(dC,1e-20)
            else:
                at=np.array([(sl[ix[i]]-om)/sg for i in range(mu)]).T
                C=(1-c1-cmu)*C+c1*(np.outer(pc,pc)+(1-hs)*cc*(2-cc)*C)+cmu*(at*w)@at.T
            sg*=np.exp((cs/ds)*(nps/chi-1)); sg=min(sg,np.mean(ranges)*2); g+=1
            if bl<pbl-1e-10: stag=0; pbl=bl
            else: stag+=1
        rc+=1

    # Phase 3: Pattern search refinement
    if best_x is not None:
        xc=best_x.copy(); fc=best; step=ranges*0.01
        while time_left()>0.05:
            imp=False
            for i in range(dim):
                if time_left()<=0.05: return best
                for s in [step[i], -step[i]]:
                    xt=xc.copy(); xt[i]+=s; ft=eval_f(np.clip(xt,lower,upper))
                    if ft<fc: xc=np.clip(xt,lower,upper); fc=ft; imp=True; break
            if not imp:
                step*=0.5
                if np.max(step/ranges)<1e-12: break
    return best
