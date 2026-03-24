import random
import math
import time


def run(func, dim, bounds, max_time):
    """
    Improved time-bounded, derivative-free minimizer (self-contained).

    What is improved vs your current best (LBFGS+ES hybrid shown):
      1) Better global-to-local schedule:
         - structured init (scrambled Halton + opposition)
         - then repeated "basin" cycles: pick elite -> do a short trust-region local search
           -> if stuck, do a principled restart (elite crossover / random / halton)
      2) Stronger local search without heavy FD gradients:
         - a bounded, coordinate/paired-coordinate pattern search with adaptive step (trust region)
         - cheap and robust under noise / non-smoothness; uses very few extra evaluations
      3) Smarter exploitation around elites:
         - elite pool with diversity filter (grid hash) to avoid many near-duplicates
         - elite "crossover" seeds (interpolation + jitter) to jump between good basins
      4) Still keeps occasional heavy-tail exploration (Cauchy) but under control.

    Returns:
        best (float): best fitness found within time limit
    """
    t0 = time.time()
    deadline = t0 + float(max_time)

    # ---------- edge cases ----------
    if dim <= 0:
        try:
            return float(func([]))
        except Exception:
            return float("inf")

    lows = [float(b[0]) for b in bounds]
    highs = [float(b[1]) for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]
    fixed = [spans[i] == 0.0 for i in range(dim)]
    span_max = max(spans) if spans else 0.0

    def now():
        return time.time()

    def clamp(x):
        y = list(x)
        for i in range(dim):
            lo = lows[i]
            hi = highs[i]
            if y[i] < lo:
                y[i] = lo
            elif y[i] > hi:
                y[i] = hi
        return y

    def evalf(x):
        return float(func(x))

    def rand_point():
        x = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                x[i] = lows[i]
            else:
                x[i] = lows[i] + random.random() * spans[i]
        return x

    def opposition(x):
        y = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                y[i] = lows[i]
            else:
                y[i] = lows[i] + highs[i] - x[i]
        return y

    def cauchy():
        u = random.random()
        return math.tan(math.pi * (u - 0.5))

    # ---------- Halton (scrambled) ----------
    def first_primes(k):
        primes = []
        c = 2
        while len(primes) < k:
            is_p = True
            r = int(c ** 0.5)
            for p in primes:
                if p > r:
                    break
                if c % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(c)
            c += 1
        return primes

    bases = first_primes(max(1, dim))
    scr = [random.random() for _ in range(dim)]

    def vdc(n, base):
        v = 0.0
        denom = 1.0
        while n:
            n, rem = divmod(n, base)
            denom *= base
            v += rem / denom
        return v

    def halton(idx):
        x = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                x[i] = lows[i]
            else:
                u = (vdc(idx + 1, bases[i]) + scr[i]) % 1.0
                x[i] = lows[i] + u * spans[i]
        return x

    # ---------- elite pool with simple diversity filter ----------
    # Keep best points but avoid storing many almost identical ones.
    elite_k = max(12, min(40, 14 + dim))
    elites = []  # list of (f, x)
    # grid quantization for "hashing" points (scale by bounds)
    # (coarse but fast; helps keep diversity)
    q_levels = 16 if dim <= 30 else 10
    seen_cells = set()

    def cell_key(x):
        # quantize each coordinate into q_levels bins
        # fixed dims ignored
        key = []
        for i in range(dim):
            if fixed[i]:
                key.append(0)
                continue
            s = spans[i]
            if s <= 0.0:
                key.append(0)
            else:
                u = (x[i] - lows[i]) / s
                b = int(u * q_levels)
                if b < 0:
                    b = 0
                elif b >= q_levels:
                    b = q_levels - 1
                key.append(b)
        return tuple(key)

    def push_elite(f, x):
        nonlocal elites
        ck = cell_key(x)
        # allow multiple per cell only if clearly better than existing ones
        # (quick check: if cell already represented and f isn't very good, skip)
        if ck in seen_cells and len(elites) >= elite_k // 2:
            # still accept if it's close to best
            if elites and f > elites[0][0] * 1.02 + 1e-12:
                return
        elites.append((f, x[:]))
        elites.sort(key=lambda t: t[0])
        # rebuild cell set from kept elites
        if len(elites) > elite_k:
            elites = elites[:elite_k]
        seen_cells.clear()
        for ff, xx in elites:
            seen_cells.add(cell_key(xx))

    # ---------- initialization ----------
    best = float("inf")
    best_x = None
    hidx = 0

    # more init for coverage, but time-safe
    init_n = max(120, min(2400, 200 + 60 * dim))
    for j in range(init_n):
        if now() >= deadline:
            return best

        if j % 17 == 0:
            x = rand_point()
        else:
            x = halton(hidx)
            hidx += 1
        x = clamp(x)
        fx = evalf(x)
        if fx < best:
            best, best_x = fx, x[:]
        push_elite(fx, x)

        # opposition sample
        if now() >= deadline:
            return best
        xo = opposition(x)
        fo = evalf(xo)
        if fo < best:
            best, best_x = fo, xo[:]
        push_elite(fo, xo)

    if best_x is None:
        best_x = rand_point()
        best = evalf(best_x)
        push_elite(best, best_x)

    # ---------- Local search: trust-region pattern search ----------
    # Uses coordinate and random-pair directions; adapts step (delta) based on success.
    # Very robust and uses few evaluations per iteration.
    eps_floor = 1e-14 + 1e-12 * (span_max + 1.0)

    def local_search(x0, f0, time_end):
        x = x0[:]
        fx = f0

        # initial trust radius: moderate, shrink over time
        frac = (now() - t0) / max(1e-12, float(max_time))
        base = (0.18 * (1.0 - 0.85 * frac) + 0.02)
        delta = [0.0] * dim
        for i in range(dim):
            if fixed[i]:
                delta[i] = 0.0
            else:
                delta[i] = max(eps_floor, base * spans[i])

        # choose a subset of dims for huge dim to keep it cheap
        if dim <= 60:
            active = list(range(dim))
        else:
            m = max(24, min(80, 12 + dim // 6))
            active = random.sample(range(dim), m)

        # helper: try move along a direction (sparse)
        def try_move(xc, fxc, idxs, steps):
            xn = xc[:]
            for k in range(len(idxs)):
                i = idxs[k]
                if fixed[i]:
                    continue
                xn[i] += steps[k]
            xn = clamp(xn)
            fn = evalf(xn)
            return fn, xn

        improved_any = False
        # cap iterations by time; each iter uses ~2*|active| evals worst case, but we stop early
        it = 0
        while now() < time_end:
            it += 1
            improved = False

            # 1) coordinate pattern
            random.shuffle(active)
            for i in active:
                if now() >= time_end:
                    break
                if fixed[i]:
                    continue
                di = delta[i]
                if di <= eps_floor:
                    continue

                # + step
                fn, xn = try_move(x, fx, [i], [di])
                if fn < fx:
                    x, fx = xn, fn
                    improved = True
                    improved_any = True
                    # expand a bit on success
                    delta[i] = min(0.45 * spans[i], max(delta[i], di) * 1.25)
                    continue

                if now() >= time_end:
                    break
                # - step
                fn, xn = try_move(x, fx, [i], [-di])
                if fn < fx:
                    x, fx = xn, fn
                    improved = True
                    improved_any = True
                    delta[i] = min(0.45 * spans[i], max(delta[i], di) * 1.25)
                else:
                    # shrink if both fail
                    delta[i] = max(eps_floor, di * 0.70)

            if now() >= time_end:
                break

            # 2) paired directions (cheap second-order-ish behavior)
            # do a few random pairs
            pair_tries = 2 if dim <= 40 else 1
            for _ in range(pair_tries):
                if now() >= time_end:
                    break
                if len(active) < 2:
                    break
                i, j = random.sample(active, 2)
                if fixed[i] and fixed[j]:
                    continue
                si = delta[i] if not fixed[i] else 0.0
                sj = delta[j] if not fixed[j] else 0.0
                if si <= eps_floor and sj <= eps_floor:
                    continue

                # try four sign combos but stop early on success
                combos = [(si, sj), (si, -sj), (-si, sj), (-si, -sj)]
                random.shuffle(combos)
                for a, b in combos:
                    if now() >= time_end:
                        break
                    fn, xn = try_move(x, fx, [i, j], [a, b])
                    if fn < fx:
                        x, fx = xn, fn
                        improved = True
                        improved_any = True
                        if not fixed[i]:
                            delta[i] = min(0.45 * spans[i], max(delta[i], abs(a)) * 1.15)
                        if not fixed[j]:
                            delta[j] = min(0.45 * spans[j], max(delta[j], abs(b)) * 1.15)
                        break

            # stop if no improvement and steps are tiny
            if not improved:
                # if most active deltas already very small, stop
                small = 0
                for i in active:
                    if fixed[i]:
                        continue
                    if delta[i] <= eps_floor * 8.0:
                        small += 1
                if small >= max(1, int(0.7 * len(active))):
                    break

            # keep iteration count modest
            if it >= (6 if dim <= 40 else 4):
                break

        return fx, x, improved_any

    # ---------- main basin-hopping loop ----------
    x_cur = best_x[:]
    f_cur = best

    no_improve = 0
    # restart threshold increases with dim
    restart_after = 90 + 16 * dim

    # schedule local search slices
    next_local = now()
    while now() < deadline:
        t = now()
        frac = (t - t0) / max(1e-12, float(max_time))

        # --- local search slice ---
        if t >= next_local:
            slice_len = 0.03 + 0.0018 * dim
            # allow a bit more early
            slice_len *= (1.15 - 0.35 * frac)
            time_end = min(deadline, t + slice_len)

            # pick seed: best elite usually, else crossover
            if elites and random.random() < 0.80:
                seed = elites[0][1][:]
            elif len(elites) >= 2 and random.random() < 0.85:
                # elite crossover seed (interpolate then jitter)
                a = 0.25 + 0.5 * random.random()
                x1 = elites[0][1]
                x2 = random.choice(elites[1:])[1]
                seed = [0.0] * dim
                for i in range(dim):
                    if fixed[i]:
                        seed[i] = lows[i]
                    else:
                        v = a * x1[i] + (1.0 - a) * x2[i]
                        # small jitter
                        v += (random.random() * 2.0 - 1.0) * (0.02 + 0.03 * (1.0 - frac)) * spans[i]
                        seed[i] = v
                seed = clamp(seed)
            else:
                seed = x_cur[:]

            fseed = evalf(seed)
            push_elite(fseed, seed)
            if fseed < best:
                best, best_x = fseed, seed[:]
                x_cur, f_cur = seed[:], fseed

            fL, xL, _ = local_search(seed, fseed, time_end)
            push_elite(fL, xL)
            if fL < best:
                best, best_x = fL, xL[:]
                x_cur, f_cur = xL[:], fL
                no_improve = 0
            else:
                no_improve += 1

            # schedule next local slice (more frequent early)
            next_local = now() + (0.05 + 0.004 * dim) * (0.35 + 0.95 * frac)

            if now() >= deadline:
                break

        # --- exploration step (cheap) ---
        # pick a base from elites or current
        if elites and random.random() < 0.30:
            x_base = random.choice(elites)[1][:]
        else:
            x_base = x_cur[:]

        # step size scale decreases with time
        scale = (0.14 * (1.0 - 0.88 * frac) + 0.010)

        r = random.random()
        if r < 0.70:
            # gaussian-ish small perturbation
            x_new = x_base[:]
            for i in range(dim):
                if fixed[i]:
                    continue
                # approx normal with 6 uniforms
                g = (random.random() + random.random() + random.random() +
                     random.random() + random.random() + random.random() - 3.0) / 3.0
                x_new[i] += g * scale * spans[i]
        elif r < 0.90:
            # heavy-tail jump
            x_new = x_base[:]
            for i in range(dim):
                if fixed[i]:
                    continue
                x_new[i] += 0.40 * scale * spans[i] * cauchy()
        else:
            # global sample
            if random.random() < 0.75:
                x_new = halton(hidx)
                hidx += 1
            else:
                x_new = rand_point()

        x_new = clamp(x_new)
        f_new = evalf(x_new)
        push_elite(f_new, x_new)

        if f_new < best:
            best, best_x = f_new, x_new[:]
            x_cur, f_cur = x_new[:], f_new
            no_improve = 0
        else:
            # accept occasionally if it's not too much worse (very mild SA-like)
            if f_new <= f_cur or random.random() < 0.03:
                x_cur, f_cur = x_new[:], f_new
            no_improve += 1

        # --- stagnation restart ---
        if no_improve >= restart_after:
            if elites and random.random() < 0.92:
                # jump to an elite, then jitter
                seed = (elites[0][1] if random.random() < 0.65 else random.choice(elites)[1])[:]
            else:
                seed = halton(hidx)
                hidx += 1

            jitter = (0.10 * (1.0 - 0.85 * frac) + 0.012)
            for i in range(dim):
                if fixed[i]:
                    continue
                seed[i] += (random.random() * 2.0 - 1.0) * jitter * spans[i]
            seed = clamp(seed)
            fseed = evalf(seed)
            push_elite(fseed, seed)

            x_cur, f_cur = seed[:], fseed
            if fseed < best:
                best, best_x = fseed, seed[:]
            no_improve = 0

    return best
