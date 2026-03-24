import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained, no externals).

    Upgrades vs your best pattern-search hybrid:
      1) Better global coverage: scrambled Halton + a bit of random.
      2) Stronger local exploitation: (1+λ)-ES with 1/5 success rule (scale adapts).
      3) Fast deterministic squeeze: opportunistic coordinate/pattern polishing.
      4) Robust escapes: occasional heavy-tailed jumps + restarts.
      5) Strict time guarding: checks inside expensive loops.

    Returns:
      best (float): best fitness found within max_time.
    """
    t0 = time.time()
    deadline = t0 + max_time

    def time_left():
        return deadline - time.time()

    # --- bounds ---
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for i in range(dim):
        if spans[i] <= 0.0:
            spans[i] = 1.0

    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def eval_f(x):
        return float(func(x))

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # --- gaussian (Box-Muller, cached) ---
    _spare = [None]
    def gauss():
        z = _spare[0]
        if z is not None:
            _spare[0] = None
            return z
        u1 = max(1e-12, random.random())
        u2 = random.random()
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        z0 = r * math.cos(th)
        z1 = r * math.sin(th)
        _spare[0] = z1
        return z0

    # Cauchy-like heavy tail: ratio of Gaussians
    def heavy_step(scale):
        g = gauss()
        h = gauss()
        return (g / max(1e-12, abs(h))) * scale

    # --- Halton (scrambled via per-dim digit permutation) ---
    def first_primes(n):
        primes = []
        k = 2
        while len(primes) < n:
            ok = True
            r = int(k ** 0.5)
            for p in primes:
                if p > r:
                    break
                if k % p == 0:
                    ok = False
                    break
            if ok:
                primes.append(k)
            k += 1
        return primes

    primes = first_primes(max(1, dim))

    # digit permutations for scrambling (different per dimension/base)
    digit_perm = {}
    for j in range(dim):
        base = primes[j]
        perm = list(range(base))
        random.shuffle(perm)
        digit_perm[(j, base)] = perm

    def halton_scrambled(index, base, perm):
        # radical inverse with digit permutation (scramble)
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            d = i % base
            r += f * perm[d]
            i //= base
        return r

    def halton_point(k):
        x = [0.0] * dim
        for j in range(dim):
            base = primes[j]
            u = halton_scrambled(k, base, digit_perm[(j, base)])
            # u is in [0,1) but with perm digits can slightly bias; clamp for safety
            if u < 0.0:
                u = 0.0
            elif u >= 1.0:
                u = math.nextafter(1.0, 0.0) if hasattr(math, "nextafter") else 1.0 - 1e-16
            x[j] = lows[j] + u * spans[j]
        return x

    # --- init incumbent ---
    best_x = rand_vec()
    best = eval_f(best_x)
    if time_left() <= 0:
        return best

    # --- global seeding: scrambled Halton + random sprinkling ---
    # modest but effective; scales with dim
    H = max(24, min(450, 30 + 14 * dim))
    R = max(8,  min(80,  6 +  3 * dim))

    # some randoms first (helps if func is weird/discontinuous)
    for _ in range(R):
        if time_left() <= 0:
            return best
        x = rand_vec()
        f = eval_f(x)
        if f < best:
            best, best_x = f, x

    for k in range(1, H + 1):
        if time_left() <= 0:
            return best
        x = halton_point(k)
        f = eval_f(x)
        if f < best:
            best, best_x = f, x

    # --- elite pool (small) ---
    elite_x = [best_x[:]]
    elite_f = [best]

    def elite_key(x):
        # coarse key for "different enough" points
        # uses normalized bins; cheap and stable
        key = []
        for i in range(dim):
            s = spans[i]
            if s <= 0:
                key.append(0)
            else:
                key.append(int(((x[i] - lows[i]) / s) * 2000.0))
        return tuple(key)

    elite_keys = {elite_key(best_x)}

    def push_elite(x, f):
        k = elite_key(x)
        if k not in elite_keys:
            elite_keys.add(k)
            elite_x.append(x[:])
            elite_f.append(f)
        # keep top 6
        idx = sorted(range(len(elite_f)), key=lambda i: elite_f[i])
        elite_x[:] = [elite_x[i] for i in idx[:6]]
        elite_f[:] = [elite_f[i] for i in idx[:6]]
        # rebuild key set for trimmed list
        elite_keys.clear()
        for ex in elite_x:
            elite_keys.add(elite_key(ex))

    # --- coordinate polish (opportunistic) ---
    eps = 1e-12 * (1.0 + (max(spans) if dim else 1.0))

    def coord_polish(x0, f0, step_frac, rounds=1):
        x = x0[:]
        f = f0
        steps = [step_frac * spans[i] for i in range(dim)]
        for _ in range(rounds):
            coords = list(range(dim))
            random.shuffle(coords)
            improved = False
            for j in coords:
                if time_left() <= 0:
                    return x, f
                sj = steps[j]
                if sj <= 0.0:
                    continue

                xp = x[:]
                xp[j] += sj
                clip_inplace(xp)
                fp = eval_f(xp)
                if fp + eps < f:
                    x, f = xp, fp
                    improved = True
                    continue

                xm = x[:]
                xm[j] -= sj
                clip_inplace(xm)
                fm = eval_f(xm)
                if fm + eps < f:
                    x, f = xm, fm
                    improved = True
                    continue

            if not improved:
                for j in range(dim):
                    steps[j] *= 0.6
        return x, f

    # --- main optimizer: (1+λ)-ES + occasional heavy-tail + polish/restarts ---
    # sigma is relative to spans (normalized)
    sigma = 0.18
    sigma_min = 1e-14
    sigma_max = 0.65

    lam = max(10, min(60, 6 + 3 * dim))

    succ = 0
    trials = 0
    since_improve = 0

    # restart control
    halton_idx = H + 1

    while time_left() > 0:
        # parent selection: mostly best, sometimes elite/recombined
        r = random.random()
        if len(elite_x) >= 2 and r < 0.18:
            a = random.randrange(len(elite_x))
            b = random.randrange(len(elite_x))
            if a == b:
                parent = elite_x[a][:]
            else:
                w = random.random()
                pa, pb = elite_x[a], elite_x[b]
                parent = [w * pa[i] + (1.0 - w) * pb[i] for i in range(dim)]
        elif r < 0.35:
            parent = elite_x[random.randrange(len(elite_x))][:]
        else:
            parent = best_x[:]

        batch_best_x = None
        batch_best_f = float("inf")

        # occasionally use heavy-tailed mutations for escaping basins
        heavy = (random.random() < 0.12)

        # generate λ offspring
        for _ in range(lam):
            if time_left() <= 0:
                return best

            x = parent[:]
            if heavy:
                for i in range(dim):
                    x[i] += heavy_step(sigma * spans[i])
            else:
                for i in range(dim):
                    x[i] += gauss() * (sigma * spans[i])

            clip_inplace(x)
            f = eval_f(x)

            if f < batch_best_f:
                batch_best_f = f
                batch_best_x = x

        # selection
        trials += 1
        if batch_best_f + eps < best:
            best = batch_best_f
            best_x = batch_best_x
            push_elite(best_x, best)
            succ += 1
            since_improve = 0
        else:
            since_improve += 1

        # 1/5 success rule adaptation (every 12 generations)
        if trials >= 12:
            rate = succ / float(trials)
            if rate > 0.2:
                sigma *= 1.22
            else:
                sigma *= 0.84
            if sigma < sigma_min:
                sigma = sigma_min
            elif sigma > sigma_max:
                sigma = sigma_max
            succ = 0
            trials = 0

        # quick polish sometimes (cheap squeeze)
        if since_improve in (6, 12) and time_left() > 0:
            bx, bf = coord_polish(best_x, best, step_frac=0.03, rounds=1)
            if bf + eps < best:
                best, best_x = bf, bx
                push_elite(best_x, best)
                since_improve = 0

        # stuck handling: inject global + reset sigma a bit
        if since_improve >= 22 and time_left() > 0:
            since_improve = 0

            # inject: 2 Halton points + 1 random + 1 heavy jump around best
            for _ in range(2):
                if time_left() <= 0:
                    return best
                xh = halton_point(halton_idx)
                halton_idx += 1
                fh = eval_f(xh)
                if fh + eps < best:
                    best, best_x = fh, xh
                    push_elite(best_x, best)

            if time_left() <= 0:
                return best
            xr = rand_vec()
            fr = eval_f(xr)
            if fr + eps < best:
                best, best_x = fr, xr
                push_elite(best_x, best)

            if time_left() <= 0:
                return best
            xj = best_x[:]
            for i in range(dim):
                xj[i] += heavy_step(max(0.05 * spans[i], sigma * spans[i]))
            clip_inplace(xj)
            fj = eval_f(xj)
            if fj + eps < best:
                best, best_x = fj, xj
                push_elite(best_x, best)

            # re-widen a bit to help escape local minima
            sigma = max(sigma, 0.14)

    return best
