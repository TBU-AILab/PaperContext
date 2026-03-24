import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (self-contained).

    Main improvements over the previous pattern-search hybrid:
      1) Better global coverage: Halton low-discrepancy sampling (+ a few randoms).
      2) Stronger local exploitation: bounded Nelder–Mead with restarts.
      3) Robust escape from local minima: heavy-tailed (Cauchy-like) perturbation.
      4) Cheap coordinate polishing to squeeze extra gains late in the run.

    Returns:
      best (float): best fitness found within max_time
    """
    t0 = time.time()
    deadline = t0 + max_time

    def time_left():
        return deadline - time.time()

    # --- Bounds / scaling ---
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

    # --- Gaussian + heavy-tail helpers (no external libs) ---
    spare = [None]
    def gauss():
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

    def cauchy_like(scale):
        # ratio of Gaussians => Cauchy-like heavy tail
        g = gauss()
        h = gauss()
        return (g / max(1e-12, abs(h))) * scale

    # --- Halton sequence (global sampling) ---
    def first_primes(n):
        primes = []
        k = 2
        while len(primes) < n:
            is_p = True
            r = int(k ** 0.5)
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

    primes = first_primes(max(1, dim))

    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def halton_point(k):
        x = [0.0] * dim
        for j in range(dim):
            u = halton(k, primes[j])
            x[j] = lows[j] + u * spans[j]
        return x

    # --- Start: pick incumbent from a mix of random + Halton ---
    best_x = rand_vec()
    best = eval_f(best_x)
    if time_left() <= 0:
        return best

    # Global sampling budget (small but effective)
    H = max(24, min(300, 30 + 12 * dim))
    # sprinkle a few random points too (helps when Halton aligns poorly)
    R = max(6, min(40, 4 + 2 * dim))

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

    # --- Nelder–Mead local search (bounded) with periodic restarts ---
    # NM parameters
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho   = 0.5   # contraction
    sigma = 0.5   # shrink

    # Build simplex around a center
    def make_simplex(center, scale_frac):
        simplex = [center[:]]
        fvals = [eval_f(center)]
        for i in range(dim):
            if time_left() <= 0:
                break
            x = center[:]
            x[i] += scale_frac * spans[i]
            clip_inplace(x)
            simplex.append(x)
            fvals.append(eval_f(x))
        return simplex, fvals

    # Coordinate polish (cheap)
    def coord_polish(x0, f0, step_frac, tries=1):
        xbest = x0[:]
        fbest = f0
        step = [step_frac * spans[i] for i in range(dim)]
        for _ in range(tries):
            coords = list(range(dim))
            random.shuffle(coords)
            improved = False
            for j in coords:
                if time_left() <= 0:
                    return xbest, fbest
                sj = step[j]
                if sj <= 0.0:
                    continue

                xp = xbest[:]
                xp[j] += sj
                clip_inplace(xp)
                fp = eval_f(xp)
                if fp < fbest:
                    xbest, fbest = xp, fp
                    improved = True
                    continue

                xm = xbest[:]
                xm[j] -= sj
                clip_inplace(xm)
                fm = eval_f(xm)
                if fm < fbest:
                    xbest, fbest = xm, fm
                    improved = True
                    continue

            # small adaptive shrink if no gain
            if not improved:
                for j in range(dim):
                    step[j] *= 0.6
        return xbest, fbest

    # Initialization
    simplex, fvals = make_simplex(best_x, 0.10)
    if fvals and fvals[0] < best:
        best, best_x = fvals[0], simplex[0][:]

    no_improve_iters = 0
    last_best = best
    halton_idx = H + 1

    while time_left() > 0:
        # Sort simplex
        order = sorted(range(len(fvals)), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        if fvals[0] < best:
            best, best_x = fvals[0], simplex[0][:]

        # centroid of best dim points (excluding worst)
        centroid = [0.0] * dim
        for i in range(dim):
            s = 0.0
            for j in range(dim):  # 0..dim-1
                s += simplex[j][i]
            centroid[i] = s / float(dim)

        worst = simplex[-1]
        f_worst = fvals[-1]
        f_second = fvals[-2]

        # Reflection
        xr = [centroid[i] + alpha * (centroid[i] - worst[i]) for i in range(dim)]
        clip_inplace(xr)
        fr = eval_f(xr)
        if time_left() <= 0:
            return best

        if fr < fvals[0]:
            # Expansion
            xe = [centroid[i] + gamma * (xr[i] - centroid[i]) for i in range(dim)]
            clip_inplace(xe)
            fe = eval_f(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        elif fr < f_second:
            simplex[-1], fvals[-1] = xr, fr
        else:
            # Contraction
            if fr < f_worst:
                xc = [centroid[i] + rho * (xr[i] - centroid[i]) for i in range(dim)]
            else:
                xc = [centroid[i] - rho * (centroid[i] - worst[i]) for i in range(dim)]
            clip_inplace(xc)
            fc = eval_f(xc)
            if fc < min(f_worst, fr):
                simplex[-1], fvals[-1] = xc, fc
            else:
                # Shrink
                xbest = simplex[0]
                for j in range(1, len(simplex)):
                    xj = simplex[j]
                    for i in range(dim):
                        xj[i] = xbest[i] + sigma * (xj[i] - xbest[i])
                    clip_inplace(xj)
                    fvals[j] = eval_f(xj)

        # Progress/stall tracking
        if best + 1e-15 < last_best:
            last_best = best
            no_improve_iters = 0
        else:
            no_improve_iters += 1

        # Light coordinate polish occasionally (helps on ridges/box constraints)
        if no_improve_iters in (10, 20) and time_left() > 0:
            bx, bf = coord_polish(best_x, best, step_frac=0.03, tries=1)
            if bf < best:
                best, best_x = bf, bx
                simplex, fvals = make_simplex(best_x, 0.08)
                last_best = best
                no_improve_iters = 0

        # Restart if stuck: heavy-tailed jump + Halton injection, then rebuild simplex
        if no_improve_iters >= 30 and time_left() > 0:
            no_improve_iters = 0

            # Candidate 1: heavy-tailed around current best
            x1 = best_x[:]
            for i in range(dim):
                scale = max(0.02 * spans[i], 1e-12 * (1.0 + spans[i]))
                x1[i] += cauchy_like(scale)
            clip_inplace(x1)
            f1 = eval_f(x1)
            if f1 < best:
                best, best_x = f1, x1

            if time_left() <= 0:
                return best

            # Candidate 2: one Halton point (global injection)
            x2 = halton_point(halton_idx)
            halton_idx += 1
            f2 = eval_f(x2)
            if f2 < best:
                best, best_x = f2, x2

            # Candidate 3: a random point (occasionally beats Halton on some problems)
            if time_left() > 0:
                x3 = rand_vec()
                f3 = eval_f(x3)
                if f3 < best:
                    best, best_x = f3, x3

            # Rebuild simplex around best with a refreshed scale
            simplex, fvals = make_simplex(best_x, 0.12)
            last_best = best

    return best
