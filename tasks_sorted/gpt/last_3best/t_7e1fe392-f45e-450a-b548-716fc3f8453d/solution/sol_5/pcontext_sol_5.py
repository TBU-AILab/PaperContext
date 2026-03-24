import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Main upgrade: a stronger hybrid that usually beats plain ES / coordinate search:
      - Global: scrambled Halton + small LHS batches + opposition points
      - Model-free local: multi-start SPAN (Simultaneous Perturbation) gradient
        descent with Adam-style steps (very evaluation-efficient in higher dim)
      - Occasional local simplex refinement (small Nelder–Mead budget) on best
      - Elite archive + stagnation-triggered restarts

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---------- helpers ----------
    spans = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    scales = [s if s > 0.0 else 1.0 for s in spans]
    inv_scales = [1.0 / s for s in scales]

    def clamp(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

    def reflect(v, lo, hi):
        # fold into [lo, hi]
        if lo == hi:
            return lo
        w = hi - lo
        if w <= 0.0:
            return lo
        z = (v - lo) % (2.0 * w)
        if z > w:
            z = 2.0 * w - z
        return lo + z

    def fix_bounds(x):
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            y[i] = reflect(x[i], lo, hi)
        return y

    def evaluate(x):
        try:
            v = func(x)
            if v is None:
                return float("inf")
            v = float(v)
            if not math.isfinite(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_uniform():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def opposite(x):
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            y[i] = clamp(lo + hi - x[i], lo, hi)
        return y

    def dist_norm_l1(a, b):
        s = 0.0
        for i in range(dim):
            s += abs(a[i] - b[i]) * inv_scales[i]
        return s

    def elite_insert(elites, x, fx, max_elites, dedup_eps=1e-10):
        elites.append((fx, x))
        elites.sort(key=lambda t: t[0])
        out = []
        for f, p in elites:
            ok = True
            for _, q in out:
                if dist_norm_l1(p, q) < dedup_eps:
                    ok = False
                    break
            if ok:
                out.append((f, p))
            if len(out) >= max_elites:
                break
        return out

    # ---------- quasi-random: scrambled Halton ----------
    def first_primes(n):
        ps = []
        x = 2
        while len(ps) < n:
            ok = True
            r = int(x ** 0.5)
            for p in ps:
                if p > r: break
                if x % p == 0:
                    ok = False
                    break
            if ok:
                ps.append(x)
            x += 1
        return ps

    primes = first_primes(max(1, dim))
    digit_perm = {}
    for b in primes:
        perm = list(range(b))
        random.shuffle(perm)
        digit_perm[b] = perm

    def halton_scrambled(index, base):
        f = 1.0
        r = 0.0
        i = index
        perm = digit_perm[base]
        while i > 0:
            f /= base
            d = i % base
            r += f * perm[d]
            i //= base
        return r

    def halton_point(k):
        x = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            u = halton_scrambled(k, primes[i])
            x[i] = lo + u * (hi - lo)
        return x

    # ---------- LHS (small batches for diversity) ----------
    def lhs_batch(n):
        per_dim = []
        for i in range(dim):
            arr = [(k + random.random()) / n for k in range(n)]
            random.shuffle(arr)
            per_dim.append(arr)
        pts = []
        for k in range(n):
            x = [0.0] * dim
            for i in range(dim):
                lo, hi = bounds[i]
                u = per_dim[i][k]
                x[i] = lo + u * (hi - lo)
            pts.append(x)
        return pts

    # ---------- local: SPSA + Adam (evaluation efficient) ----------
    def spsa_adam(x0, f0, steps_budget, step_frac, delta_frac, iters):
        """
        SPSA gradient estimate uses 2 evals/iter regardless of dim:
            g_i ≈ (f(x+c*Δ)-f(x-c*Δ)) / (2*c*Δ_i)
        Then Adam update with per-dimension moments.
        """
        x = list(x0)
        fx = f0

        # Adam state
        m = [0.0] * dim
        v = [0.0] * dim
        beta1, beta2 = 0.9, 0.999
        eps = 1e-12

        # base sizes
        a = step_frac
        c = delta_frac

        t = 0
        evals_left = steps_budget

        while evals_left >= 2 and t < iters and time.time() < deadline:
            t += 1

            # random +/-1 perturbation vector
            Delta = [1.0 if random.getrandbits(1) else -1.0 for _ in range(dim)]

            # perturbation magnitude per dim
            ck = c / (t ** 0.101)  # slow decay
            x_plus = [0.0] * dim
            x_minus = [0.0] * dim
            for i in range(dim):
                ci = ck * scales[i]
                x_plus[i] = reflect(x[i] + ci * Delta[i], bounds[i][0], bounds[i][1])
                x_minus[i] = reflect(x[i] - ci * Delta[i], bounds[i][0], bounds[i][1])

            f_plus = evaluate(x_plus)
            f_minus = evaluate(x_minus)
            evals_left -= 2

            if not math.isfinite(f_plus): f_plus = float("inf")
            if not math.isfinite(f_minus): f_minus = float("inf")
            if not (math.isfinite(f_plus) and math.isfinite(f_minus)):
                # if objective is nasty here, just stop this local run
                break

            # gradient estimate
            denom_base = 2.0 * ck
            g = [0.0] * dim
            diff = (f_plus - f_minus)
            for i in range(dim):
                # denom = 2*ck*scale_i * Delta_i  -> using denom_base*scale_i
                denom = denom_base * scales[i]
                if denom <= 0.0:
                    continue
                g[i] = diff / (denom * Delta[i])

            # Adam step size schedule (mild decay)
            ak = a / (t ** 0.602)

            # update Adam moments and parameters
            for i in range(dim):
                gi = g[i]
                m[i] = beta1 * m[i] + (1.0 - beta1) * gi
                v[i] = beta2 * v[i] + (1.0 - beta2) * (gi * gi)

                # bias correction
                mhat = m[i] / (1.0 - (beta1 ** t))
                vhat = v[i] / (1.0 - (beta2 ** t))

                step_i = ak * scales[i] * (mhat / (math.sqrt(vhat) + eps))
                # gradient descent
                x[i] = reflect(x[i] - step_i, bounds[i][0], bounds[i][1])

            f_new = evaluate(x)
            evals_left -= 1
            if evals_left < 0:
                break

            if f_new < fx:
                fx = f_new
            else:
                # small backoff if not improving
                a *= 0.98

        return x, fx, (steps_budget - evals_left)

    # ---------- local: small Nelder–Mead polish (cheap) ----------
    def nelder_mead(x0, f0, budget_evals, init_step_frac):
        if budget_evals <= 0:
            return x0, f0, 0

        # build simplex
        simplex = [list(x0)]
        fvals = [f0]
        used = 0
        step = [max(1e-12 * scales[i], init_step_frac * scales[i]) for i in range(dim)]
        for i in range(dim):
            if used >= budget_evals or time.time() >= deadline:
                break
            xi = list(x0)
            xi[i] = reflect(xi[i] + step[i], bounds[i][0], bounds[i][1])
            fi = evaluate(xi)
            simplex.append(xi)
            fvals.append(fi)
            used += 1

        m = len(simplex)
        if m < 2:
            return x0, f0, used

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        while used < budget_evals and time.time() < deadline:
            idx = sorted(range(m), key=lambda i: fvals[i])
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            bestx, bestf = simplex[0], fvals[0]
            worstx, worstf = simplex[-1], fvals[-1]
            second_worstf = fvals[-2]

            centroid = [0.0] * dim
            for j in range(m - 1):
                xj = simplex[j]
                for d in range(dim):
                    centroid[d] += xj[d]
            inv = 1.0 / (m - 1)
            for d in range(dim):
                centroid[d] *= inv

            # reflect
            xr = [reflect(centroid[d] + alpha * (centroid[d] - worstx[d]),
                          bounds[d][0], bounds[d][1]) for d in range(dim)]
            fr = evaluate(xr); used += 1
            if fr < bestf and used < budget_evals:
                # expand
                xe = [reflect(centroid[d] + gamma * (xr[d] - centroid[d]),
                              bounds[d][0], bounds[d][1]) for d in range(dim)]
                fe = evaluate(xe); used += 1
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            elif fr < second_worstf:
                simplex[-1], fvals[-1] = xr, fr
            else:
                # contract
                if fr < worstf:
                    xc = [reflect(centroid[d] + rho * (xr[d] - centroid[d]),
                                  bounds[d][0], bounds[d][1]) for d in range(dim)]
                else:
                    xc = [reflect(centroid[d] - rho * (centroid[d] - worstx[d]),
                                  bounds[d][0], bounds[d][1]) for d in range(dim)]
                fc = evaluate(xc); used += 1
                if fc < worstf:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    # shrink
                    for j in range(1, m):
                        if used >= budget_evals or time.time() >= deadline:
                            break
                        xs = [reflect(bestx[d] + sigma * (simplex[j][d] - bestx[d]),
                                      bounds[d][0], bounds[d][1]) for d in range(dim)]
                        fs = evaluate(xs); used += 1
                        simplex[j], fvals[j] = xs, fs

        bi = min(range(len(fvals)), key=lambda i: fvals[i])
        return simplex[bi], fvals[bi], used

    # ---------- initialization ----------
    best = float("inf")
    best_x = None
    elites = []
    max_elites = max(12, min(80, 5 * dim))

    # global initial points: Halton + opposition + a little LHS
    n_hal = max(24, min(160, 10 * dim))
    hal_start = 1 + random.randrange(256)
    for k in range(n_hal):
        if time.time() >= deadline:
            return best if best_x is not None else float("inf")
        x = halton_point(hal_start + k)
        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x
        elites = elite_insert(elites, x, fx, max_elites)

        if (k & 1) == 0:
            xo = opposite(x)
            fxo = evaluate(xo)
            if fxo < best:
                best, best_x = fxo, xo
            elites = elite_insert(elites, xo, fxo, max_elites)

    n_lhs = max(8, min(60, 3 * dim))
    for x in lhs_batch(n_lhs):
        if time.time() >= deadline:
            return best if best_x is not None else float("inf")
        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x
        elites = elite_insert(elites, x, fx, max_elites)

    if best_x is None:
        best_x = rand_uniform()
        best = evaluate(best_x)
        elites = [(best, best_x)]

    # ---------- main loop ----------
    phase = 0
    no_improve = 0

    while time.time() < deadline:
        phase += 1
        m = len(elites) if elites else 1
        if not elites:
            elites = [(best, best_x)]

        # pick elite (best-biased)
        r = random.random()
        idx = int((r * r) * m)
        fx0, x0 = elites[idx]

        # SPSA parameters depend on rank/stagnation
        rank = idx / max(1, m - 1)
        stagn = 1.0 + min(3.0, no_improve / 15.0)

        # step and delta fractions (relative to spans)
        # bigger for worse elites + if stagnating
        step_frac = (0.02 + 0.10 * rank) * stagn
        delta_frac = (0.01 + 0.06 * rank) * (1.0 + 0.3 * stagn)

        # evaluation budgets (time-safe)
        # SPSA: ~ (2 + 1) evals per iter (two-sided + eval after step)
        spsa_iters = 6 + (dim // 3)
        spsa_iters = min(spsa_iters, 30)
        spsa_budget = min(3 * spsa_iters + 3, 120)

        x1, f1, _ = spsa_adam(x0, fx0, spsa_budget, step_frac, delta_frac, spsa_iters)
        elites = elite_insert(elites, x1, f1, max_elites)

        if f1 < best:
            best, best_x = f1, x1
            no_improve = 0
        else:
            no_improve += 1

        if time.time() >= deadline:
            break

        # occasional Nelder–Mead polish on best (helps when near a basin)
        if phase % 4 == 0:
            nm_budget = min(40 + 3 * dim, 180)
            x2, f2, _ = nelder_mead(best_x, best, nm_budget, init_step_frac=0.06)
            elites = elite_insert(elites, x2, f2, max_elites)
            if f2 < best:
                best, best_x = f2, x2
                no_improve = 0

        if time.time() >= deadline:
            break

        # global injections / restarts
        if phase % 9 == 0:
            # one fresh Halton + opposition
            k = hal_start + n_hal + phase * 3 + random.randrange(8)
            xg = halton_point(k)
            fg = evaluate(xg)
            elites = elite_insert(elites, xg, fg, max_elites)
            if fg < best:
                best, best_x = fg, xg
                no_improve = 0

            xo = opposite(xg)
            fo = evaluate(xo)
            elites = elite_insert(elites, xo, fo, max_elites)
            if fo < best:
                best, best_x = fo, xo
                no_improve = 0

        # if very stuck: random restart + small LHS burst
        if no_improve >= 25 and time.time() < deadline:
            xr = rand_uniform()
            fr = evaluate(xr)
            elites = elite_insert(elites, xr, fr, max_elites)
            if fr < best:
                best, best_x = fr, xr

            for x in lhs_batch(max(6, min(20, 2 * dim))):
                if time.time() >= deadline:
                    break
                fx = evaluate(x)
                elites = elite_insert(elites, x, fx, max_elites)
                if fx < best:
                    best, best_x = fx, x
            no_improve = 0

    return best
