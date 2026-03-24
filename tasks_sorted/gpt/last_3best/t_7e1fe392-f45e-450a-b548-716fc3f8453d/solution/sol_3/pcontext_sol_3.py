import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (no external libs).

    Main changes vs previous:
      - Stronger global exploration: scrambled Halton + opposition points.
      - Better local search: Powell-style pattern search (coordinate + acceleration).
      - Adaptive per-dimension step sizes with success/failure tracking.
      - Trust-region-like shrinking + periodic multi-start from elites.

    Returns:
        best (float): best objective value found within max_time
    """
    t0 = time.time()
    deadline = t0 + max_time

    # ---------- basic helpers ----------
    spans = [bounds[i][1] - bounds[i][0] for i in range(dim)]
    inv_spans = [1.0 / s if s != 0.0 else 0.0 for s in spans]

    def isfinite(x):
        return math.isfinite(x)

    def clamp(v, lo, hi):
        if v < lo: return lo
        if v > hi: return hi
        return v

    def reflect(v, lo, hi):
        # reflection tends to preserve distribution better than clamp
        if lo == hi:
            return lo
        if v < lo:
            v = lo + (lo - v)
            if v > hi:
                # fold into range
                v = lo + (v - lo) % (hi - lo)
        elif v > hi:
            v = hi - (v - hi)
            if v < lo:
                v = lo + (v - lo) % (hi - lo)
        return clamp(v, lo, hi)

    def evaluate(x):
        try:
            v = func(x)
            if v is None:
                return float("inf")
            v = float(v)
            if not isfinite(v):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_uniform():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def opposite(x):
        # opposition-based point: lo+hi-x
        y = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            y[i] = clamp(lo + hi - x[i], lo, hi)
        return y

    # ---------- Halton (scrambled-ish via random digit permutation) ----------
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

    primes = first_primes(max(1, dim))
    # digit permutations per base for a light Owen-style scramble (cheap)
    digit_perm = {}
    for b in primes:
        perm = list(range(b))
        random.shuffle(perm)
        digit_perm[b] = perm

    def halton_scrambled(index, base):
        # index starts at 1
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

    # ---------- elite archive ----------
    def norm_l1(a, b):
        s = 0.0
        for i in range(dim):
            if spans[i] != 0.0:
                s += abs(a[i] - b[i]) * inv_spans[i]
        return s

    def elite_insert(elites, x, fx, max_elites):
        elites.append((fx, x))
        elites.sort(key=lambda t: t[0])
        out = []
        # keep diverse enough points
        for f, p in elites:
            keep = True
            for _, q in out:
                if norm_l1(p, q) < 1e-8:
                    keep = False
                    break
            if keep:
                out.append((f, p))
            if len(out) >= max_elites:
                break
        return out

    # ---------- Powell-style pattern search ----------
    def pattern_search(x0, f0, steps, max_sweeps):
        """
        Coordinate exploratory moves, then a pattern (acceleration) move.
        steps: per-dimension absolute step sizes.
        """
        x = list(x0)
        fx = f0

        min_steps = []
        for i in range(dim):
            s = spans[i]
            # minimum step tied to scale; avoid total stagnation
            min_steps.append(max(1e-15, (s * 1e-12) if s > 0 else 1e-12))

        # exploratory multipliers: try full, half; cheap but effective
        mults = (1.0, 0.5)

        for _ in range(max_sweeps):
            if time.time() >= deadline:
                break

            x_prev = list(x)
            f_prev = fx

            improved_any = False
            order = list(range(dim))
            random.shuffle(order)

            for i in order:
                if time.time() >= deadline:
                    break
                if spans[i] == 0.0:
                    continue

                lo, hi = bounds[i]
                si = max(steps[i], min_steps[i])

                best_i_val = x[i]
                best_i_f = fx

                base = x[i]
                for m in mults:
                    d = si * m
                    # try both directions
                    v1 = base + d
                    v2 = base - d
                    if v1 >= lo and v1 <= hi:
                        xt = list(x); xt[i] = v1
                        ft = evaluate(xt)
                        if ft < best_i_f:
                            best_i_f = ft
                            best_i_val = v1
                    if v2 >= lo and v2 <= hi:
                        xt = list(x); xt[i] = v2
                        ft = evaluate(xt)
                        if ft < best_i_f:
                            best_i_f = ft
                            best_i_val = v2

                if best_i_f < fx:
                    x[i] = best_i_val
                    fx = best_i_f
                    improved_any = True
                    # a bit of step growth for that coordinate
                    steps[i] = min(spans[i] if spans[i] > 0 else steps[i], steps[i] * 1.15)
                else:
                    # small decay for that coordinate
                    steps[i] = max(min_steps[i], steps[i] * 0.85)

            if time.time() >= deadline:
                break

            # pattern/acceleration move (Powell-ish): x + (x - x_prev)
            if improved_any:
                xp = [0.0] * dim
                for i in range(dim):
                    lo, hi = bounds[i]
                    xp[i] = reflect(x[i] + (x[i] - x_prev[i]), lo, hi)
                fp = evaluate(xp)
                if fp < fx:
                    x, fx = xp, fp
                    # reward: slightly increase all steps
                    for i in range(dim):
                        steps[i] = min(spans[i] if spans[i] > 0 else steps[i], steps[i] * 1.1)
                else:
                    # mild contraction if pattern failed
                    for i in range(dim):
                        steps[i] = max(min_steps[i], steps[i] * 0.9)
            else:
                # no improvement in a whole sweep -> global shrink
                for i in range(dim):
                    steps[i] = max(min_steps[i], steps[i] * 0.6)
                # stopping condition: all steps tiny
                tiny = True
                for i in range(dim):
                    if steps[i] > min_steps[i] * 20.0:
                        tiny = False
                        break
                if tiny:
                    break

            # if we didn't improve at all, break early
            if fx >= f_prev and not improved_any:
                break

        return x, fx, steps

    # ---------- initialization (stronger global coverage) ----------
    best = float("inf")
    best_x = None
    elites = []
    max_elites = max(10, min(60, 4 * dim))

    # time-aware initial sampling size
    n_init = max(24, min(140, 10 * dim))
    halton_start = 1 + random.randrange(128)

    for k in range(n_init):
        if time.time() >= deadline:
            return best if best_x is not None else float("inf")

        # mix: Halton, random, and opposition of each
        if k % 5 != 4:
            x = halton_point(halton_start + k)
        else:
            x = rand_uniform()

        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x
        elites = elite_insert(elites, x, fx, max_elites)

        xo = opposite(x)
        fxo = evaluate(xo)
        if fxo < best:
            best, best_x = fxo, xo
        elites = elite_insert(elites, xo, fxo, max_elites)

    if best_x is None:
        best_x = rand_uniform()
        best = evaluate(best_x)
        elites = [(best, best_x)]

    # base steps for local search
    base_steps = []
    for i in range(dim):
        s = spans[i]
        base_steps.append(0.25 * s if s > 0 else 1.0)

    # ---------- main loop: multi-start local search + perturbation ----------
    phase = 0
    no_improve = 0

    while time.time() < deadline:
        phase += 1

        # choose a start point: best-biased elite selection
        m = len(elites)
        if m == 0:
            elites = [(best, best_x)]
            m = 1
        r = random.random()
        idx = int((r * r) * m)  # quadratic bias towards best
        fx0, x0 = elites[idx]

        # steps scaled by rank + a gentle time decay
        decay = 1.0 / (1.0 + 0.02 * phase)
        rank = idx / max(1, m - 1)
        steps = [max(1e-15, base_steps[i] * (0.15 + 1.35 * rank) * decay) for i in range(dim)]

        # local refinement
        sweeps = 2 + (dim // 3)
        sweeps = min(sweeps, 12)
        x1, f1, _ = pattern_search(x0, fx0, steps, max_sweeps=sweeps)

        if f1 < best:
            best, best_x = f1, x1
            no_improve = 0
        else:
            no_improve += 1

        elites = elite_insert(elites, x1, f1, max_elites)

        if time.time() >= deadline:
            break

        # perturbation around best (small Gaussian), stronger if stagnating
        stagnation_boost = 1.0 + min(3.0, no_improve / 10.0)
        xg = [0.0] * dim
        for i in range(dim):
            lo, hi = bounds[i]
            s = spans[i] if spans[i] > 0 else 1.0
            sigma = (0.02 + 0.08 * rank) * s * decay * stagnation_boost
            xg[i] = reflect(best_x[i] + random.gauss(0.0, sigma), lo, hi)
        fg = evaluate(xg)

        if fg < best:
            best, best_x = fg, xg
            no_improve = 0
        elites = elite_insert(elites, xg, fg, max_elites)

        # periodic global injection (prevents over-localization)
        if phase % 8 == 0 and time.time() < deadline:
            xh = halton_point(halton_start + n_init + phase * 3)
            fh = evaluate(xh)
            if fh < best:
                best, best_x = fh, xh
                no_improve = 0
            elites = elite_insert(elites, xh, fh, max_elites)

            xho = opposite(xh)
            fho = evaluate(xho)
            if fho < best:
                best, best_x = fho, xho
                no_improve = 0
            elites = elite_insert(elites, xho, fho, max_elites)

        # if very stuck, do a bigger restart
        if no_improve >= 25 and time.time() < deadline:
            xr = rand_uniform()
            fr = evaluate(xr)
            if fr < best:
                best, best_x = fr, xr
            elites = elite_insert(elites, xr, fr, max_elites)
            no_improve = 0

    return best
