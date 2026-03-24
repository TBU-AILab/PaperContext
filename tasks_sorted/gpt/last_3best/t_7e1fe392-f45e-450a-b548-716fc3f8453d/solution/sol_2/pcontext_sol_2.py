import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimizer (self-contained, no external libs).

    Hybrid:
      - Low-discrepancy global exploration (Halton sequence + a few random points)
      - Elite archive
      - Local refinement: (1+lambda) Evolution Strategy with step-size control
      - Occasional coordinate/pattern polishing around the current best
      - Soft restarts

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---------- helpers ----------
    spans = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    inv_spans = [1.0 / s if s != 0 else 0.0 for s in spans]

    def clamp(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

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

    # --- Halton sequence (simple, fast, good coverage) ---
    def first_primes(n):
        ps = []
        x = 2
        while len(ps) < n:
            ok = True
            r = int(x**0.5)
            for p in ps:
                if p > r: break
                if x % p == 0:
                    ok = False
                    break
            if ok:
                ps.append(x)
            x += 1
        return ps

    primes = first_primes(dim)

    def halton_single(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k):
        # k should start at 1
        x = []
        for i in range(dim):
            u = halton_single(k, primes[i])
            lo, hi = bounds[i]
            x.append(lo + u * (hi - lo))
        return x

    # distance in normalized L1 (cheap)
    def norm_l1(a, b):
        s = 0.0
        for i in range(dim):
            if spans[i] != 0:
                s += abs(a[i] - b[i]) * inv_spans[i]
        return s

    # elite maintenance
    def elite_insert(elites, x, fx, max_elites):
        elites.append((fx, x))
        elites.sort(key=lambda t: t[0])
        out = []
        for f, p in elites:
            keep = True
            for f2, p2 in out:
                if norm_l1(p, p2) < 1e-10:
                    keep = False
                    break
            if keep:
                out.append((f, p))
            if len(out) >= max_elites:
                break
        return out

    # --- ES mutation with reflection at bounds ---
    def mutate_gauss(x, sigmas):
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            v = x[i] + random.gauss(0.0, sigmas[i])

            # reflect if out of bounds (often better than clamp for ES)
            if v < lo:
                v = lo + (lo - v)
                if v > hi:
                    v = lo
            elif v > hi:
                v = hi - (v - hi)
                if v < lo:
                    v = hi
            y[i] = v
        return y

    # --- coordinate polish around best (few evaluations, good for separable-ish problems) ---
    def coordinate_polish(x0, f0, step_fracs, rounds):
        x = list(x0)
        fx = f0
        for _ in range(rounds):
            if time.time() >= deadline:
                break
            improved = False
            order = list(range(dim))
            random.shuffle(order)
            for i in order:
                if time.time() >= deadline:
                    break
                lo, hi = bounds[i]
                span = spans[i]
                if span == 0:
                    continue
                step = step_fracs[i] * span
                if step <= 0:
                    continue
                base = x[i]
                best_i = base
                best_f = fx
                # try a small pattern: +/- step, +/- step/2
                for d in (step, -step, 0.5 * step, -0.5 * step):
                    v = base + d
                    if v < lo or v > hi:
                        continue
                    xt = list(x)
                    xt[i] = v
                    ft = evaluate(xt)
                    if ft < best_f:
                        best_f = ft
                        best_i = v
                if best_f < fx:
                    x[i] = best_i
                    fx = best_f
                    improved = True
            if not improved:
                break
        return x, fx

    # ---------- initialization ----------
    best = float("inf")
    best_x = None
    elites = []

    # initial budget: mixture of Halton + random (robust on many dims)
    n_init = max(16, min(80, 6 * dim))
    halton_start = 1 + random.randrange(32)  # randomize start index a bit

    for k in range(n_init):
        if time.time() >= deadline:
            return best if best_x is not None else float("inf")
        if k < int(0.8 * n_init):
            x = halton_point(halton_start + k)
        else:
            x = rand_uniform()
        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x
        elites = elite_insert(elites, x, fx, max_elites=max(8, min(40, 3 * dim)))

    if best_x is None:
        best_x = rand_uniform()
        best = evaluate(best_x)
        elites = [(best, best_x)]

    # base sigma per dimension
    base_sigma = []
    for i in range(dim):
        s = spans[i]
        base_sigma.append(0.2 * s if s > 0 else 1.0)

    # ES state
    sigma_scale = 0.5   # global multiplier
    succ_ema = 0.2      # success rate EMA
    succ_target = 0.2   # target for 1/5-ish rule
    max_elites = max(8, min(40, 3 * dim))

    it = 0
    while time.time() < deadline:
        it += 1

        # choose parent: mostly best, sometimes other elites (maintains diversity)
        if elites and random.random() < 0.75:
            # bias to better elites
            m = len(elites)
            r = random.random()
            idx = int((r * r) * m)
            parent_f, parent_x = elites[idx]
        else:
            parent_x = best_x
            parent_f = best

        # ES offspring count (small and time-safe)
        lam = 4 + (dim // 4)
        if lam > 24:
            lam = 24

        # build per-dim sigmas: shrink around best, larger for worse parents
        # rank-like scaling without explicit rank:
        parent_quality = 1.0
        if parent_f > best and math.isfinite(best) and abs(best) + 1.0 > 0:
            parent_quality = 1.0 + 0.25 * min(4.0, (parent_f - best) / (abs(best) + 1.0))
        sigmas = [max(1e-15, base_sigma[i] * sigma_scale * parent_quality) for i in range(dim)]

        # generate offspring and pick best
        best_off_f = float("inf")
        best_off_x = None
        for _ in range(lam):
            if time.time() >= deadline:
                break
            x = mutate_gauss(parent_x, sigmas)
            fx = evaluate(x)
            if fx < best_off_f:
                best_off_f, best_off_x = fx, x

        if best_off_x is None:
            break

        # selection / archive
        improved = (best_off_f < parent_f)
        succ_ema = 0.9 * succ_ema + 0.1 * (1.0 if improved else 0.0)

        elites = elite_insert(elites, best_off_x, best_off_f, max_elites=max_elites)
        if best_off_f < best:
            best, best_x = best_off_f, best_off_x

        # step-size control (smooth 1/5-style)
        # if success > target -> increase, else decrease
        if succ_ema > succ_target:
            sigma_scale *= 1.08
        else:
            sigma_scale *= 0.93

        # keep sigma_scale within reasonable bounds
        if sigma_scale < 1e-6:
            sigma_scale = 1e-6
        if sigma_scale > 2.0:
            sigma_scale = 2.0

        # occasional coordinate polish on current best (cheap extra gain)
        if it % 10 == 0 and time.time() < deadline:
            step_fracs = []
            # polish step tied to current sigma scale (smaller over time)
            for i in range(dim):
                if spans[i] == 0:
                    step_fracs.append(0.0)
                else:
                    # between 0.1% and 2% of span
                    frac = 0.002 + 0.02 * min(1.0, sigma_scale)
                    step_fracs.append(frac)
            x2, f2 = coordinate_polish(best_x, best, step_fracs, rounds=1)
            if f2 < best:
                best, best_x = f2, x2
            elites = elite_insert(elites, x2, f2, max_elites=max_elites)

        # soft restart if stagnating: inject a fresh global sample
        if it % 25 == 0 and time.time() < deadline:
            xg = halton_point(halton_start + n_init + it)
            fg = evaluate(xg)
            if fg < best:
                best, best_x = fg, xg
            elites = elite_insert(elites, xg, fg, max_elites=max_elites)

            # if no improvement for long, bump exploration a bit
            if succ_ema < 0.05:
                sigma_scale = min(2.0, sigma_scale * 1.3)

    return best
