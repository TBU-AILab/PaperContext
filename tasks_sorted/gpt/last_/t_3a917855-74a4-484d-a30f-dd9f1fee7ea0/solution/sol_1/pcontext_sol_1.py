import random
import math
import time

def run(func, dim, bounds, max_time):
    start = time.time()
    deadline = start + max_time

    # ---------- helpers ----------
    def clamp(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def ensure_bounds(x):
        for i in range(dim):
            lo, hi = bounds[i]
            x[i] = clamp(x[i], lo, hi)
        return x

    def eval_f(x):
        return float(func(x))

    span = []
    for i in range(dim):
        lo, hi = bounds[i]
        s = hi - lo
        span.append(s if s > 0 else 1.0)

    def rand_vec():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    # Halton for decent initial coverage
    def first_primes(n):
        primes = []
        k = 2
        while len(primes) < n:
            is_p = True
            r = int(math.isqrt(k))
            for p in primes:
                if p > r:
                    break
                if k % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(k)
            k += 1
        return primes

    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    primes = first_primes(dim)

    def halton_vec(k):
        x = []
        for i in range(dim):
            u = halton(k, primes[i])
            lo, hi = bounds[i]
            x.append(lo + u * (hi - lo))
        return x

    # reflection keeps steps meaningful near the boundary
    def reflect_1d(v, lo, hi):
        if lo >= hi:
            return lo
        w = hi - lo
        t = (v - lo) % (2.0 * w)
        return (lo + t) if t <= w else (hi - (t - w))

    # ---------- incumbent ----------
    best_x = rand_vec()
    best = eval_f(best_x)

    # ---------- parameters ----------
    # global sampling budget (time-safe): a bit more than before
    halton_budget = max(40, 25 * dim)
    k = 1

    # ES / local search
    lam = 10 if dim <= 8 else (16 if dim <= 25 else 22)
    sigma = 0.25          # relative to span
    sigma_min = 1e-14
    sigma_max = 0.60

    # coordinate/pattern search
    coord_period = 7      # run coordinate pass every N ES iters
    base_coord_step = 0.10  # relative to span, will anneal with sigma

    # restarts
    stall = 0
    stall_limit = 45 if dim <= 15 else 65

    # lightweight "archive" to restart from good places (avoid single-basin fixation)
    archive = []          # list of (f, x)
    archive_cap = 6

    def push_archive(fx, x):
        nonlocal archive
        archive.append((fx, x[:]))
        archive.sort(key=lambda t: t[0])
        # remove near-duplicates (cheap)
        pruned = []
        for f, v in archive:
            ok = True
            for f2, v2 in pruned:
                # L1 distance normalized
                d = 0.0
                for i in range(dim):
                    d += abs(v[i] - v2[i]) / span[i]
                if d < 1e-3 * dim:
                    ok = False
                    break
            if ok:
                pruned.append((f, v))
            if len(pruned) >= archive_cap:
                break
        archive = pruned

    push_archive(best, best_x)

    # ---------- coordinate search around a center ----------
    def coord_search(center_x, center_f, step_rel):
        # Opportunistic +/- coordinate moves with shrinking step.
        x = center_x[:]
        f = center_f
        idx = list(range(dim))
        random.shuffle(idx)
        for i in idx:
            lo, hi = bounds[i]
            step = step_rel * span[i]
            if step <= 0:
                continue

            orig = x[i]

            # try + step
            v = reflect_1d(orig + step, lo, hi)
            if v != orig:
                x[i] = v
                fp = eval_f(x)
            else:
                fp = float('inf')

            # try - step
            v2 = reflect_1d(orig - step, lo, hi)
            if v2 != orig:
                x[i] = v2
                fm = eval_f(x)
            else:
                fm = float('inf')

            # pick best move (or revert)
            if fp < f and fp <= fm:
                f = fp
                # keep x[i] as currently set (v2 or v?) — ensure:
                x[i] = reflect_1d(orig + step, lo, hi)
            elif fm < f:
                f = fm
                x[i] = reflect_1d(orig - step, lo, hi)
            else:
                x[i] = orig

            if time.time() >= deadline:
                break
        return f, x

    # ---------- main loop ----------
    es_iter = 0
    while True:
        if time.time() >= deadline:
            return best

        # 1) global exploration (Halton)
        if k <= halton_budget:
            x = halton_vec(k)
            k += 1
            f = eval_f(x)
            if f < best:
                best, best_x = f, x
                stall = 0
                push_archive(best, best_x)
            else:
                stall += 1
            continue

        es_iter += 1

        # 2) local search: (1+λ)-ES, but keep the best offspring and accept if better
        best_off_f = best
        best_off_x = None
        successes = 0

        # occasional: sample a couple of candidates from archive as parents (diversity)
        if archive and (es_iter % 9 == 0):
            parent_f, parent_x = random.choice(archive)
        else:
            parent_f, parent_x = best, best_x

        for _ in range(lam):
            if time.time() >= deadline:
                return best

            cand = parent_x[:]
            # gaussian perturbation
            for i in range(dim):
                lo, hi = bounds[i]
                step = random.gauss(0.0, sigma) * span[i]
                cand[i] = reflect_1d(cand[i] + step, lo, hi)

            f = eval_f(cand)
            if f < best_off_f:
                best_off_f = f
                best_off_x = cand
            if f < parent_f:
                successes += 1

        if best_off_x is not None and best_off_f < best:
            best, best_x = best_off_f, best_off_x
            push_archive(best, best_x)
            stall = 0
        else:
            stall += 1

        # adapt sigma using a success-rate heuristic (more stable than always scaling on single improvement)
        # target ~0.2 success rate
        sr = successes / float(lam)
        if sr > 0.22:
            sigma = min(sigma_max, sigma * 1.12)
        elif sr < 0.18:
            sigma = max(sigma_min, sigma * 0.88)

        # 3) coordinate / pattern search refinement (good for separable / sharp valleys)
        if es_iter % coord_period == 0:
            step_rel = max(1e-6, min(0.25, base_coord_step * (sigma / 0.25)))
            f2, x2 = coord_search(best_x, best, step_rel)
            if f2 < best:
                best, best_x = f2, x2
                push_archive(best, best_x)
                stall = 0

        # 4) restart if stuck (use archive-best + random mixture)
        if stall >= stall_limit:
            stall = 0
            sigma = 0.25

            # choose a seed: best from archive or the incumbent
            if archive and random.random() < 0.7:
                seed_f, seed_x = archive[0]
            else:
                seed_f, seed_x = best, best_x

            # mix with a random point; sometimes do a strong jump
            r = rand_vec()
            mix = 0.15 if random.random() < 0.5 else 0.45
            x = [mix * seed_x[i] + (1.0 - mix) * r[i] for i in range(dim)]
            ensure_bounds(x)
            f = eval_f(x)
            if f < best:
                best, best_x = f, x
                push_archive(best, best_x)

            # also: short Halton "micro-burst" after restart (cheap diversification)
            burst = min(10, max_time)  # time-independent iteration count
            for _ in range(burst):
                if time.time() >= deadline:
                    return best
                xh = halton_vec(k)
                k += 1
                fh = eval_f(xh)
                if fh < best:
                    best, best_x = fh, xh
                    push_archive(best, best_x)

    # return best  (unreachable)
