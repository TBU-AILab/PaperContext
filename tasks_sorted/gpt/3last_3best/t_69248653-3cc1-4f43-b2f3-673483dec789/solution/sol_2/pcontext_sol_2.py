import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded minimizer (stdlib-only): multi-start + bandit-selected local solvers.

    Core ideas vs. your current ASA:
    - Better initialization: lightweight low-discrepancy (Halton) global seeding.
    - Multi-start portfolio: maintains multiple "incumbents" and improves them in parallel.
    - Bandit scheduling (UCB1): automatically spends more evaluations on the most promising solver type.
    - Robust local refinement: (1+1)-ES with 1/5th success rule + coordinate search + occasional Cauchy jumps.
    - Automatic restarts when a track stagnates; also keeps a small elite set.

    Returns: best objective value found within max_time.
    """

    t0 = time.time()
    deadline = t0 + float(max_time)

    # ----------------- utilities -----------------
    lo = [float(b[0]) for b in bounds]
    hi = [float(b[1]) for b in bounds]
    span = [hi[i] - lo[i] for i in range(dim)]
    scale = [(span[i] if span[i] > 0 else 1.0) for i in range(dim)]

    def clip_inplace(x):
        for i in range(dim):
            if x[i] < lo[i]:
                x[i] = lo[i]
            elif x[i] > hi[i]:
                x[i] = hi[i]
        return x

    def safe_eval(x):
        try:
            v = func(x)
            if v is None or isinstance(v, complex):
                return float("inf")
            v = float(v)
            if v != v or v == float("inf") or v == -float("inf"):
                return float("inf")
            return v
        except Exception:
            return float("inf")

    def rand_point():
        return [random.uniform(lo[i], hi[i]) for i in range(dim)]

    # ----------------- Halton sequence (low-discrepancy) -----------------
    def _primes_upto(n):
        if n < 2:
            return []
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for p in range(2, int(n**0.5) + 1):
            if sieve[p]:
                step = p
                start = p * p
                sieve[start:n+1:step] = [False] * (((n - start) // step) + 1)
        return [i for i, isprime in enumerate(sieve) if isprime]

    def _first_n_primes(n):
        # enough upper bound for small/medium dim; expand if needed
        # for n=200, 1223 is enough; for safety we scale.
        ub = max(50, int(n * (math.log(max(3, n)) + math.log(math.log(max(3, n))) + 3)))
        primes = _primes_upto(ub)
        while len(primes) < n:
            ub = int(ub * 1.6) + 10
            primes = _primes_upto(ub)
        return primes[:n]

    primes = _first_n_primes(dim)

    def halton_value(index, base):
        # radical inverse
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k, scramble):
        # k starts at 1 for standard Halton
        x = [0.0] * dim
        for i in range(dim):
            u = halton_value(k, primes[i])
            # Cranley-Patterson rotation for scrambling
            u = (u + scramble[i]) % 1.0
            x[i] = lo[i] + u * span[i]
        return x

    # ----------------- local move operators -----------------
    def cauchy():
        # standard Cauchy via tan(pi*(u-0.5)), clamp extremes to avoid inf
        u = random.random()
        # avoid exactly 0 or 1
        u = min(1.0 - 1e-12, max(1e-12, u))
        return math.tan(math.pi * (u - 0.5))

    # ----------------- track (incumbent) state -----------------
    class Track:
        __slots__ = ("x", "fx", "sigma", "succ", "tries", "stagn", "best_fx")
        def __init__(self, x, fx, sigma):
            self.x = x
            self.fx = fx
            self.best_fx = fx
            self.sigma = sigma  # global step scale (multiplies per-dim scale)
            self.succ = 0
            self.tries = 0
            self.stagn = 0

    # ----------------- bandit arms (operators) -----------------
    # We apply an operator to a chosen track and return improvement (positive if improved global best).
    # Arms:
    # 0: (1+1)-ES Gaussian
    # 1: (1+1)-ES Cauchy (heavy tail)
    # 2: coordinate pattern search
    # 3: local random restart around elite
    N_ARMS = 4
    arm_plays = [1e-9] * N_ARMS
    arm_reward = [0.0] * N_ARMS

    def ucb_select(total):
        # UCB1: maximize avg_reward + sqrt(2 ln t / n)
        best_arm = 0
        best_val = -1e300
        ln_t = math.log(max(2.0, total))
        for a in range(N_ARMS):
            avg = arm_reward[a] / arm_plays[a]
            bonus = math.sqrt(2.0 * ln_t / arm_plays[a])
            val = avg + bonus
            if val > best_val:
                best_val = val
                best_arm = a
        return best_arm

    # ----------------- seeding: Halton + random -----------------
    # Use small time fraction but cap by evaluations.
    scramble = [random.random() for _ in range(dim)]
    seed_n = max(30, 12 * dim)
    # tracks count: a few parallel incumbents
    n_tracks = max(3, min(10, 2 + dim // 3))

    seeds = []
    # Mix halton and random to reduce structure risk
    for k in range(1, seed_n + 1):
        if time.time() >= deadline:
            break
        if k % 4 != 0:
            x = halton_point(k, scramble)
        else:
            x = rand_point()
        fx = safe_eval(x)
        seeds.append((fx, x))

    if not seeds:
        # last resort
        x = rand_point()
        return safe_eval(x)

    seeds.sort(key=lambda t: t[0])
    best = seeds[0][0]
    best_x = list(seeds[0][1])

    # initialize tracks from best seeds; diversify by adding noise
    tracks = []
    base_sigma = 0.15  # relative to scale
    for i in range(n_tracks):
        fx, x = seeds[min(i, len(seeds) - 1)]
        xx = list(x)
        # slight perturbation for diversity
        for j in range(dim):
            if random.random() < 0.6:
                xx[j] += random.gauss(0.0, 0.03 * scale[j])
        clip_inplace(xx)
        fxx = safe_eval(xx)
        if fxx < fx:
            fx, x = fxx, xx
        sigma = base_sigma * (0.7 + 0.6 * random.random())
        tracks.append(Track(list(x), fx, sigma))
        if fx < best:
            best, best_x = fx, list(x)

    # maintain small elite set for restarts
    elite = [(best, list(best_x))]

    def update_elite(fx, x):
        nonlocal elite
        elite.append((fx, list(x)))
        elite.sort(key=lambda t: t[0])
        if len(elite) > max(4, n_tracks):
            elite = elite[:max(4, n_tracks)]

    # ----------------- main loop -----------------
    total_steps = 1
    # stagnation thresholds
    track_patience = max(80, 25 * dim)
    # minimum sigma
    sigma_min = 1e-12
    sigma_max = 0.8

    # For step-size 1/5 success rule: update every window
    window = max(20, 5 * dim)

    while time.time() < deadline:
        total_steps += 1

        # pick track: prioritize worse tracks a bit less, but still keep exploring
        # simple: choose random track, biased to better ones
        # (roulette on rank)
        tracks_sorted = sorted(range(len(tracks)), key=lambda i: tracks[i].fx)
        r = random.random()
        # rank-biased: geometric
        idx = tracks_sorted[min(len(tracks_sorted) - 1, int(math.log(1.0 / max(1e-12, 1.0 - r))))]
        tr = tracks[idx]

        arm = ucb_select(total_steps)

        old_best = best
        new_fx = None

        if arm == 0:
            # (1+1)-ES Gaussian
            xn = list(tr.x)
            for j in range(dim):
                xn[j] += random.gauss(0.0, tr.sigma * scale[j])
            clip_inplace(xn)
            fn = safe_eval(xn)

            tr.tries += 1
            if fn < tr.fx:
                tr.x, tr.fx = xn, fn
                tr.succ += 1
                tr.stagn = 0
            else:
                tr.stagn += 1

            new_fx = tr.fx

        elif arm == 1:
            # (1+1)-ES Cauchy (heavy-tailed exploration)
            xn = list(tr.x)
            for j in range(dim):
                step = tr.sigma * scale[j] * 0.6
                xn[j] += step * cauchy()
            clip_inplace(xn)
            fn = safe_eval(xn)

            tr.tries += 1
            if fn < tr.fx:
                tr.x, tr.fx = xn, fn
                tr.succ += 1
                tr.stagn = 0
            else:
                tr.stagn += 1

            new_fx = tr.fx

        elif arm == 2:
            # coordinate pattern search around track point
            xn = list(tr.x)
            # try a few coordinates with +/- steps; take best
            best_local_fx = tr.fx
            best_local_x = tr.x
            # step magnitude tied to sigma but with floor
            for _ in range(2 if dim <= 8 else 3):
                j = random.randrange(dim)
                step = max(1e-15 * scale[j], tr.sigma * scale[j])
                for sgn in (-1.0, 1.0):
                    xc = list(xn)
                    xc[j] += sgn * step
                    clip_inplace(xc)
                    fc = safe_eval(xc)
                    if fc < best_local_fx:
                        best_local_fx = fc
                        best_local_x = xc
            tr.tries += 1
            if best_local_fx < tr.fx:
                tr.x, tr.fx = list(best_local_x), best_local_fx
                tr.succ += 1
                tr.stagn = 0
            else:
                tr.stagn += 1

            new_fx = tr.fx

        else:
            # local restart around a random elite (keeps exploitation while escaping)
            ef, ex = elite[random.randrange(len(elite))]
            xn = list(ex)
            # radius depends on how stuck this track is
            widen = 1.0 + min(6.0, tr.stagn / max(1.0, track_patience / 2.0))
            for j in range(dim):
                xn[j] += random.gauss(0.0, widen * 0.25 * scale[j])
            clip_inplace(xn)
            fn = safe_eval(xn)

            # Replace the track if better than its current, or if it's very stuck
            if fn < tr.fx or tr.stagn > track_patience:
                tr.x, tr.fx = xn, fn
                tr.stagn = 0
                # do not count as success for step-size adaptation
            else:
                tr.stagn += 1

            new_fx = tr.fx

        # global best update
        if new_fx is not None and new_fx < best:
            best = new_fx
            best_x = list(tr.x)
            update_elite(best, best_x)

        # bandit reward: improvement in global best (scaled)
        imp = max(0.0, old_best - best)
        # compress reward so occasional large jumps don't dominate too much
        reward = math.log1p(imp)  # log(1+imp)
        arm_plays[arm] += 1.0
        arm_reward[arm] += reward

        # adapt sigma using 1/5 success rule per track
        if tr.tries >= window:
            rate = tr.succ / max(1, tr.tries)
            # 1/5 rule target ~0.2
            if rate > 0.22:
                tr.sigma *= 1.25
            elif rate < 0.18:
                tr.sigma *= 0.8
            # clamp
            if tr.sigma < sigma_min:
                tr.sigma = sigma_min
            elif tr.sigma > sigma_max:
                tr.sigma = sigma_max
            tr.tries = 0
            tr.succ = 0

        # restart a track if stagnating: re-seed near best or globally
        if tr.stagn > track_patience:
            if random.random() < 0.75:
                xn = list(best_x)
                for j in range(dim):
                    xn[j] += random.gauss(0.0, 0.2 * scale[j])
                clip_inplace(xn)
            else:
                # fresh global point; use Halton index based on time/steps
                k = 1 + (total_steps % max(10, 50 * dim))
                xn = halton_point(k, scramble)
            fn = safe_eval(xn)
            tr.x, tr.fx = xn, fn
            tr.stagn = 0
            tr.sigma = base_sigma * (0.7 + 0.6 * random.random())
            if fn < best:
                best = fn
                best_x = list(xn)
                update_elite(best, best_x)

    return best
