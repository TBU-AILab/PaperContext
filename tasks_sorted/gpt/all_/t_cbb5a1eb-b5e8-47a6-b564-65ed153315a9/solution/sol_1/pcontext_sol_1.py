import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimization (self-contained, no numpy).

    Main ideas vs prior version:
      - Better global exploration: low-discrepancy (Halton) sampling + a small amount of uniform random
      - Stronger local search: coordinate-wise + Gaussian mutations, with mirrored bounds handling
      - More effective restarts: restart from best with multi-scale sigma + occasional pure global
      - Cheap intensification: pattern/line search along successful directions

    Returns best (minimum) objective value found within max_time.
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---- bounds prep ----
    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    for s in spans:
        if s < 0:
            raise ValueError("Each bound must be (low, high) with low <= high.")

    # ---- helpers ----
    def evaluate(x):
        return float(func(x))

    def rand_uniform_point():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # Box-Muller normal
    def randn():
        u1 = random.random()
        u2 = random.random()
        u1 = u1 if u1 > 1e-12 else 1e-12
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # Mirror a coordinate into [lo, hi] (better than hard clipping for search)
    def mirror(v, lo, hi):
        if lo == hi:
            return lo
        w = hi - lo
        # bring into [0, 2w) then mirror
        y = (v - lo) % (2.0 * w)
        if y <= w:
            return lo + y
        else:
            return hi - (y - w)

    def mirror_point(x):
        for i in range(dim):
            x[i] = mirror(x[i], lows[i], highs[i])
        return x

    # ---- low-discrepancy global sampling (Halton) ----
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            is_p = True
            r = int(math.sqrt(x))
            for p in primes:
                if p > r:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(x)
            x += 1
        return primes

    def van_der_corput(index, base):
        # index >= 1 recommended
        vdc = 0.0
        denom = 1.0
        i = index
        while i > 0:
            denom *= base
            i, rem = divmod(i, base)
            vdc += rem / denom
        return vdc

    primes = first_primes(max(1, dim))

    def halton_point(k):
        # k starts at 1
        x = [0.0] * dim
        for i in range(dim):
            u = van_der_corput(k, primes[i])
            x[i] = lows[i] + u * spans[i]
        return x

    # ---- initialization: mixed Halton + random ----
    best = float("inf")
    best_x = None

    # Decide an initial evaluation budget based on time and dimension (safe)
    # We keep it modest and rely on ongoing local search.
    init_n = max(12, min(120, 12 + 8 * int(math.sqrt(max(1, dim)))))

    k = 1
    for j in range(init_n):
        if time.time() >= deadline:
            return best
        # mix: mostly Halton, sometimes pure random to avoid structure issues
        if random.random() < 0.85:
            x = halton_point(k)
            k += 1
        else:
            x = rand_uniform_point()
        fx = evaluate(x)
        if fx < best:
            best, best_x = fx, x[:]

    if best_x is None:
        x = rand_uniform_point()
        best = evaluate(x)
        best_x = x[:]

    # ---- local search state ----
    x = best_x[:]
    fx = best

    # Multiple step scales; start moderate
    sigma = 0.18  # relative to span
    sigma_min = 1e-10
    sigma_max = 0.75

    # Track for adaptive sigma (success rule)
    window = max(12, 6 * dim)
    succ = 0
    trials = 0

    # Stagnation / restart logic
    it_no_best = 0
    it_no_curr = 0
    # Scale with dim; higher dim -> more patience
    restart_after = 80 + 20 * dim
    hard_restart_after = 220 + 40 * dim

    # Keep last successful direction for a cheap pattern move
    last_step = [0.0] * dim
    have_step = False

    # Pre-generate a permutation list for coordinate search cycles (reshuffle occasionally)
    coords = list(range(dim))
    random.shuffle(coords)
    coord_idx = 0
    reshuffle_every = max(20, 3 * dim)
    reshuffle_count = 0

    while time.time() < deadline:
        # ---- restarts ----
        if it_no_best >= hard_restart_after:
            # strong global restart
            x = rand_uniform_point()
            fx = evaluate(x)
            if fx < best:
                best, best_x = fx, x[:]
                it_no_best = 0
            # reset local state
            sigma = 0.25
            succ = trials = 0
            it_no_curr = 0
            have_step = False
            continue

        if it_no_curr >= restart_after:
            # restart around global best at a random scale, or global random
            if random.random() < 0.75:
                x = best_x[:]
                # broaden or narrow randomly
                scale = 2.0 ** random.uniform(-1.5, 1.5)
                sigma = min(sigma_max, max(0.06, sigma * scale))
                # small random kick
                x2 = x[:]
                for i in range(dim):
                    if spans[i] > 0:
                        x2[i] += (0.25 * sigma * spans[i]) * randn()
                mirror_point(x2)
                f2 = evaluate(x2)
                if f2 <= fx:
                    x, fx = x2, f2
            else:
                x = rand_uniform_point()
                fx = evaluate(x)
                sigma = 0.3
            if fx < best:
                best, best_x = fx, x[:]
                it_no_best = 0
            it_no_curr = 0
            succ = trials = 0
            have_step = False
            continue

        # ---- choose a move type ----
        # Mixture:
        #  - coordinate move (good in box-bounded problems)
        #  - Gaussian full-dimensional move
        #  - occasional global Halton probe near remaining time
        r = random.random()

        x_new = x[:]

        if r < 0.55 and dim > 0:
            # Coordinate-wise perturbation (1-3 coordinates)
            reshuffle_count += 1
            if reshuffle_count >= reshuffle_every:
                random.shuffle(coords)
                reshuffle_count = 0
            # choose how many coordinates to change
            kchg = 1 if random.random() < 0.75 else (2 if random.random() < 0.85 else 3)
            for _ in range(min(kchg, dim)):
                i = coords[coord_idx]
                coord_idx = (coord_idx + 1) % dim
                if spans[i] > 0:
                    x_new[i] += (sigma * spans[i]) * randn()
        elif r < 0.92:
            # Full-dimensional Gaussian
            for i in range(dim):
                if spans[i] > 0:
                    x_new[i] += (sigma * spans[i]) * randn()
        else:
            # Rare global probe (Halton)
            x_new = halton_point(k)
            k += 1

        mirror_point(x_new)
        f_new = evaluate(x_new)

        trials += 1

        if f_new <= fx:
            # accept improvement to current
            step = [x_new[i] - x[i] for i in range(dim)]
            x, fx = x_new, f_new
            succ += 1
            it_no_curr = 0

            # update global best
            if f_new < best:
                best, best_x = f_new, x_new[:]
                it_no_best = 0
            else:
                it_no_best += 1

            # store direction and try a cheap pattern/line step
            last_step = step
            have_step = True

            # pattern move (only when we have a direction and time)
            if have_step and time.time() < deadline:
                # try x + alpha * step, alpha in {1, 0.5} (very cheap)
                for alpha in (1.0, 0.5):
                    xp = [x[i] + alpha * last_step[i] for i in range(dim)]
                    mirror_point(xp)
                    fp = evaluate(xp)
                    if fp < fx:
                        x, fx = xp, fp
                        if fp < best:
                            best, best_x = fp, xp[:]
                            it_no_best = 0
                        else:
                            it_no_best += 1
                        succ += 1
                        it_no_curr = 0
                        break
        else:
            it_no_curr += 1
            it_no_best += 1  # weak signal of stagnation overall

        # ---- adapt sigma ----
        if trials >= window:
            rate = succ / float(trials)
            # Target ~0.2; adjust more smoothly than before
            if rate > 0.28:
                sigma *= 1.22
            elif rate < 0.12:
                sigma *= 0.78
            else:
                # mild drift toward moderate exploration
                sigma *= 1.02 if rate > 0.2 else 0.98

            if sigma < sigma_min:
                sigma = sigma_min
            elif sigma > sigma_max:
                sigma = sigma_max

            succ = 0
            trials = 0

    return best
