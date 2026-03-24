import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded black-box minimization (no external libs).

    Improvements over provided code:
      - Uses CMA-ES as the main driver (very strong on continuous bounded problems)
      - Restarts with increasing population (IPOP) when stagnating
      - Diagonal pre-conditioning at start, full covariance adaptation thereafter
      - Boundary handling via smooth resampling + fallback reflection
      - Small, throttled final local search around best

    Returns:
        best (float): best objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max(0.0, float(max_time))

    # --- bounds / scaling ---
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] <= 0.0:
            spans[i] = 1.0

    def now():
        return time.time()

    def clamp(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def reflect_bounds(x):
        # reflection with modulo folding to handle large excursions
        for i in range(dim):
            lo, hi = lows[i], highs[i]
            if x[i] < lo or x[i] > hi:
                w = hi - lo
                if w <= 0.0:
                    x[i] = lo
                else:
                    y = x[i] - lo
                    m = y % (2.0 * w)
                    x[i] = lo + (m if m <= w else (2.0 * w - m))
            x[i] = clamp(x[i], lo, hi)
        return x

    # Box-Muller
    spare = [None]
    def randn():
        if spare[0] is not None:
            z = spare[0]
            spare[0] = None
            return z
        u1 = max(1e-12, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        spare[0] = z1
        return z0

    def dot(a, b):
        return sum(ai * bi for ai, bi in zip(a, b))

    def mat_vec(M, v):
        return [dot(row, v) for row in M]

    def outer(u, v):
        return [[ui * vj for vj in v] for ui in u]

    def mat_add(A, B, alpha=1.0):
        n = len(A)
        m = len(A[0])
        C = [[0.0] * m for _ in range(n)]
        for i in range(n):
            Ai = A[i]; Bi = B[i]; Ci = C[i]
            for j in range(m):
                Ci[j] = Ai[j] + alpha * Bi[j]
        return C

    def mat_scale(A, s):
        return [[s * aij for aij in row] for row in A]

    def eye(n):
        I = [[0.0]*n for _ in range(n)]
        for i in range(n):
            I[i][i] = 1.0
        return I

    def symmetrize(A):
        n = len(A)
        for i in range(n):
            for j in range(i+1, n):
                v = 0.5*(A[i][j] + A[j][i])
                A[i][j] = v
                A[j][i] = v
        return A

    # Jacobi eigen-decomposition for symmetric matrices
    def jacobi_eig_sym(A, max_sweeps=25):
        n = len(A)
        V = eye(n)
        A = [row[:] for row in A]
        for _ in range(max_sweeps):
            # find largest off-diagonal
            p = 0
            q = 1 if n > 1 else 0
            maxv = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    v = abs(A[i][j])
                    if v > maxv:
                        maxv = v
                        p, q = i, j
            if maxv < 1e-12:
                break
            app = A[p][p]
            aqq = A[q][q]
            apq = A[p][q]
            if abs(apq) < 1e-18:
                continue
            tau = (aqq - app) / (2.0 * apq)
            t = 1.0 / (abs(tau) + math.sqrt(1.0 + tau*tau))
            if tau < 0.0:
                t = -t
            c = 1.0 / math.sqrt(1.0 + t*t)
            s = t * c

            # rotate A
            for k in range(n):
                if k != p and k != q:
                    akp = A[k][p]
                    akq = A[k][q]
                    A[k][p] = c*akp - s*akq
                    A[p][k] = A[k][p]
                    A[k][q] = s*akp + c*akq
                    A[q][k] = A[k][q]

            app2 = c*c*app - 2.0*s*c*apq + s*s*aqq
            aqq2 = s*s*app + 2.0*s*c*apq + c*c*aqq
            A[p][p] = app2
            A[q][q] = aqq2
            A[p][q] = 0.0
            A[q][p] = 0.0

            # rotate V
            for k in range(n):
                vkp = V[k][p]
                vkq = V[k][q]
                V[k][p] = c*vkp - s*vkq
                V[k][q] = s*vkp + c*vkq

        evals = [A[i][i] for i in range(n)]
        return evals, V

    def make_BD(C):
        # eigen-decomp C = B diag(D^2) B^T, return B and D
        C = symmetrize(C)
        evals, B = jacobi_eig_sym(C, max_sweeps=20 + 2*dim)
        # clamp eigenvalues
        D = [math.sqrt(max(1e-20, ev)) for ev in evals]
        return B, D

    def BDz(B, D, z):
        # B * (D * z)
        Dz = [D[i] * z[i] for i in range(dim)]
        return mat_vec(B, Dz)

    # objective wrapper
    def eval_f(x):
        return float(func(x))

    # --- quick init: center + random probes ---
    center = [0.5*(lows[i] + highs[i]) for i in range(dim)]
    bestx = center[:]
    best = eval_f(bestx) if now() < deadline else float("inf")

    # some random samples for robustness
    for _ in range(min(10*dim, 80)):
        if now() >= deadline:
            return best
        x = [lows[i] + random.random()*spans[i] for i in range(dim)]
        fx = eval_f(x)
        if fx < best:
            best, bestx = fx, x

    if now() >= deadline:
        return best

    # --- CMA-ES with IPOP restarts ---
    # initial mean: best random found
    m = bestx[:]

    # initial sigma scaled to box
    sigma0 = 0.25
    sigma = sigma0

    # set initial covariance to diagonal based on spans (in normalized coords it becomes I)
    # we operate in normalized space y in [0,1]^d for numerical stability.
    def to_y(x):
        return [(x[i] - lows[i]) / spans[i] for i in range(dim)]

    def to_x(y):
        x = [lows[i] + y[i] * spans[i] for i in range(dim)]
        return reflect_bounds(x)

    y_m = to_y(m)
    y_best = to_y(bestx)

    # restart settings
    base_lambda = 4 + int(3 * math.log(dim + 1.0))
    lam = max(8, base_lambda)
    restart = 0

    # local search (very small) near end or on improvement
    def local_refine(x0, f0, budget):
        x = x0[:]
        fx = f0
        step = 0.03
        for _ in range(budget):
            if now() >= deadline:
                break
            d = random.randrange(dim)
            for sgn in (-1.0, 1.0):
                xt = x[:]
                xt[d] += sgn * step * spans[d]
                reflect_bounds(xt)
                ft = eval_f(xt)
                if ft < fx:
                    x, fx = xt, ft
                    break
            step *= 0.98
            if step < 1e-6:
                break
        return x, fx

    # main restart loop until time
    last_improve_time = now()
    last_best_val = best

    while now() < deadline:
        restart += 1
        # CMA params (from standard recommendations)
        mu = lam // 2
        weights = [math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)]
        wsum = sum(weights)
        weights = [w / wsum for w in weights]
        mueff = 1.0 / sum(w*w for w in weights)

        cc = (4.0 + mueff/dim) / (dim + 4.0 + 2.0*mueff/dim)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        c1 = 2.0 / ((dim + 1.3)**2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0/mueff) / ((dim + 2.0)**2 + mueff))
        damps = 1.0 + 2.0*max(0.0, math.sqrt((mueff - 1.0)/(dim + 1.0)) - 1.0) + cs

        # evolution paths
        pc = [0.0] * dim
        ps = [0.0] * dim

        # covariance in y-space
        C = eye(dim)
        B = eye(dim)
        D = [1.0] * dim
        inv_sqrt_C = eye(dim)

        # expectation of ||N(0,I)||
        chiN = math.sqrt(dim) * (1.0 - 1.0/(4.0*dim) + 1.0/(21.0*dim*dim))

        # bookkeeping
        eigeneval = 0
        counteval = 0
        best_restart = best

        # slightly bias mean to best ever (helps restarts)
        y_m = y_best[:]

        # generation loop
        while now() < deadline:
            # decompose occasionally
            if counteval - eigeneval > lam * (1.0 / (c1 + cmu) / dim / 10.0):
                eigeneval = counteval
                B, D = make_BD(C)
                # inv_sqrt_C = B * diag(1/D) * B^T
                # compute via columns of B
                inv_sqrt_C = [[0.0]*dim for _ in range(dim)]
                # temp = B * diag(1/D)
                temp = [[B[i][j] / max(1e-20, D[j]) for j in range(dim)] for i in range(dim)]
                # inv_sqrt_C = temp * B^T
                for i in range(dim):
                    for j in range(dim):
                        inv_sqrt_C[i][j] = sum(temp[i][k] * B[j][k] for k in range(dim))

            # sample population
            arz = []
            ary = []
            arx = []
            fits = []

            for _ in range(lam):
                if now() >= deadline:
                    break
                z = [randn() for _ in range(dim)]
                y = [y_m[i] + sigma * v for i, v in enumerate(BDz(B, D, z))]
                # handle bounds in y-space by resampling a few times, then clamp/reflect in x
                ok = True
                for _try in range(4):
                    ok = True
                    for i in range(dim):
                        if y[i] < 0.0 or y[i] > 1.0:
                            ok = False
                            break
                    if ok:
                        break
                    z = [randn() for _ in range(dim)]
                    y = [y_m[i] + sigma * v for i, v in enumerate(BDz(B, D, z))]
                # if still out, clamp in y then map; this is a last resort
                if not ok:
                    y = [clamp(y[i], 0.0, 1.0) for i in range(dim)]

                x = to_x(y)
                f = eval_f(x)
                counteval += 1

                arz.append(z)
                ary.append(y)
                arx.append(x)
                fits.append(f)

                if f < best:
                    best = f
                    bestx = x[:]
                    y_best = y[:]
                    last_improve_time = now()

            if not fits:
                break

            # sort by fitness
            idx = sorted(range(len(fits)), key=lambda i: fits[i])
            fits_sorted = [fits[i] for i in idx]
            ary_sorted = [ary[i] for i in idx]
            arz_sorted = [arz[i] for i in idx]

            if fits_sorted[0] < best_restart:
                best_restart = fits_sorted[0]

            # recombination for mean
            y_old = y_m[:]
            y_m = [0.0]*dim
            for i in range(mu):
                wi = weights[i]
                yi = ary_sorted[i]
                for d in range(dim):
                    y_m[d] += wi * yi[d]

            # update evolution path ps
            ydiff = [(y_m[d] - y_old[d]) / max(1e-20, sigma) for d in range(dim)]
            # inv_sqrt_C * ydiff
            invCy = mat_vec(inv_sqrt_C, ydiff)
            for d in range(dim):
                ps[d] = (1.0 - cs) * ps[d] + math.sqrt(cs * (2.0 - cs) * mueff) * invCy[d]

            ps_norm = math.sqrt(sum(v*v for v in ps))
            hsig = 1.0 if (ps_norm / math.sqrt(1.0 - (1.0 - cs)**(2.0 * counteval / max(1.0, lam))) / chiN) < (1.4 + 2.0/(dim+1.0)) else 0.0

            # update evolution path pc
            for d in range(dim):
                pc[d] = (1.0 - cc) * pc[d] + hsig * math.sqrt(cc * (2.0 - cc) * mueff) * (y_m[d] - y_old[d]) / max(1e-20, sigma)

            # rank-one and rank-mu updates
            # C = (1-c1-cmu)*C + c1*(pc pc^T + (1-hsig)*cc*(2-cc)*C) + cmu*sum(w_i * (y_i - y_old)(...)^T / sigma^2)
            C = mat_scale(C, 1.0 - c1 - cmu)
            if hsig < 0.5:
                C = mat_add(C, C, alpha=c1 * (1.0 - hsig) * cc * (2.0 - cc) / max(1e-30, (1.0 - c1 - cmu)))
                # Above is a mild correction; keep simple and stable.

            C = mat_add(C, outer(pc, pc), alpha=c1)

            # rank-mu
            for i in range(mu):
                wi = weights[i]
                dy = [(ary_sorted[i][d] - y_old[d]) / max(1e-20, sigma) for d in range(dim)]
                C = mat_add(C, outer(dy, dy), alpha=cmu * wi)

            # stabilize
            C = symmetrize(C)

            # step-size control
            sigma *= math.exp((cs / damps) * (ps_norm / chiN - 1.0))
            if sigma < 1e-12:
                sigma = 1e-12
            if sigma > 2.0:
                sigma = 2.0

            # occasional tiny local refinement when improved or late
            frac = (now() - t0) / max(1e-12, max_time)
            if (fits_sorted[0] <= best + 1e-15 and (frac > 0.7 or (now() - last_improve_time) < 0.02)) and now() < deadline:
                xr, fr = local_refine(bestx, best, budget=min(10 + dim, 40))
                if fr < best:
                    best, bestx = fr, xr[:]
                    y_best = to_y(bestx)

            # termination / restart conditions: stagnation or too small sigma
            if sigma < 1e-8:
                break
            if (now() - last_improve_time) > max(0.08, 0.20 * max_time / (restart + 0.5)):
                # no improvement for a while -> restart
                break
            if counteval > 20000 and dim > 25:
                # avoid spending too long on a single restart in high dim
                break

        # update global best bookkeeping
        if best < last_best_val - 1e-15:
            last_best_val = best

        # restart: increase population (IPOP) and reset sigma, mean near best
        lam = min(6 * (4 + int(3 * math.log(dim + 1.0))) * (2 ** min(6, restart)), 240 + 20*dim)
        sigma = sigma0 * (0.8 + 0.15 * restart)
        # set mean to best with small jitter in y space
        y_m = to_y(bestx)
        for d in range(dim):
            y_m[d] = clamp(y_m[d] + 0.02 * randn(), 0.0, 1.0)

    return float(best)
