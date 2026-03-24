import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Improved time-bounded derivative-free minimizer (no external libs).

    Strategy (hybrid, time-aware):
      1) Sobol-like coverage replacement: Halton global sampling (cheap, good space-filling)
         + opposition-based tries (reflect through center) to accelerate early progress.
      2) Lightweight Nelder-Mead (simplex) local search around the current best
         using bound-handling by clipping.
      3) If simplex stalls, switch to adaptive coordinate pattern steps + occasional
         heavy-tailed perturbation restarts.
      4) Budget/time checks everywhere.

    Returns:
      best (float): best fitness found within max_time.
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

    # --- Halton for global exploration ---
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

    # --- Random helpers ---
    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # Gaussian (Box-Muller)
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

    # Heavy-tailed step using ratio of Gaussians (Cauchy-like)
    def heavy_step(scale):
        g = gauss()
        h = gauss()
        return (g / max(1e-12, abs(h))) * scale

    # --- Initial incumbent ---
    best_x = rand_vec()
    best = eval_f(best_x)
    if time_left() <= 0:
        return best

    # --- Global initialization: Halton + opposition ---
    # Keep modest but strong; adapt to dim
    G = max(24, min(220, 30 + 10 * dim))
    center = [(lows[i] + highs[i]) * 0.5 for i in range(dim)]

    for k in range(1, G + 1):
        if time_left() <= 0:
            return best
        x = halton_point(k)
        f = eval_f(x)
        if f < best:
            best, best_x = f, x

        # Opposition-based candidate: reflect around center
        xo = [2.0 * center[i] - x[i] for i in range(dim)]
        clip_inplace(xo)
        fo = eval_f(xo)
        if fo < best:
            best, best_x = fo, xo

    # --- Build an initial simplex around best for Nelder-Mead ---
    # Simplex size: fraction of span, not too tiny
    init_scale = 0.08
    simplex = [best_x[:]]
    fvals = [best]
    for i in range(dim):
        if time_left() <= 0:
            return best
        x = best_x[:]
        x[i] += init_scale * spans[i]
        clip_inplace(x)
        simplex.append(x)
        fvals.append(eval_f(x))

    # NM parameters
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho   = 0.5   # contraction
    sigma = 0.5   # shrink

    # Stall detection
    nm_no_improve = 0
    last_best_nm = best

    # Fallback pattern search step sizes
    step = [0.20 * spans[i] for i in range(dim)]
    min_step = [1e-10 * (1.0 + spans[i]) for i in range(dim)]
    pat_no_improve = 0

    # --- Main loop: alternate NM and fallback steps with restarts ---
    while time_left() > 0:
        # Decide whether to do NM iteration or fallback
        # Use NM while it makes progress; otherwise fallback for a bit.
        do_nm = (nm_no_improve < 12)

        if do_nm:
            # Sort simplex
            order = sorted(range(len(fvals)), key=lambda i: fvals[i])
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            if fvals[0] < best:
                best = fvals[0]
                best_x = simplex[0][:]

            # centroid of best dim points (excluding worst)
            centroid = [0.0] * dim
            for i in range(dim):
                s = 0.0
                for j in range(dim):  # first dim points
                    s += simplex[j][i]
                centroid[i] = s / float(dim)

            worst = simplex[-1]
            f_worst = fvals[-1]
            second_worst = fvals[-2]

            # reflection
            xr = [centroid[i] + alpha * (centroid[i] - worst[i]) for i in range(dim)]
            clip_inplace(xr)
            fr = eval_f(xr)
            if time_left() <= 0:
                return best

            if fr < fvals[0]:
                # expansion
                xe = [centroid[i] + gamma * (xr[i] - centroid[i]) for i in range(dim)]
                clip_inplace(xe)
                fe = eval_f(xe)
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            elif fr < second_worst:
                simplex[-1], fvals[-1] = xr, fr
            else:
                # contraction
                if fr < f_worst:
                    # outside contraction
                    xc = [centroid[i] + rho * (xr[i] - centroid[i]) for i in range(dim)]
                else:
                    # inside contraction
                    xc = [centroid[i] - rho * (centroid[i] - worst[i]) for i in range(dim)]
                clip_inplace(xc)
                fc = eval_f(xc)

                if fc < min(f_worst, fr):
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    # shrink towards best
                    xbest = simplex[0]
                    for j in range(1, len(simplex)):
                        xj = simplex[j]
                        for i in range(dim):
                            xj[i] = xbest[i] + sigma * (xj[i] - xbest[i])
                        clip_inplace(xj)
                        fvals[j] = eval_f(xj)

            # Stall tracking
            if best + 1e-15 < last_best_nm:
                last_best_nm = best
                nm_no_improve = 0
            else:
                nm_no_improve += 1

            continue

        # --- Fallback: pattern + occasional heavy-tailed restart around best ---
        # Try coordinate pattern steps
        improved = False
        coords = list(range(dim))
        random.shuffle(coords)

        for j in coords:
            if time_left() <= 0:
                return best

            sj = step[j]
            if sj < min_step[j]:
                continue

            # plus
            cand = best_x[:]
            cand[j] += sj
            clip_inplace(cand)
            f1 = eval_f(cand)
            if f1 < best:
                best, best_x = f1, cand
                improved = True
                continue

            # minus
            cand = best_x[:]
            cand[j] -= sj
            clip_inplace(cand)
            f2 = eval_f(cand)
            if f2 < best:
                best, best_x = f2, cand
                improved = True
                continue

        if improved:
            pat_no_improve = 0
            # mild expand to speed up when going downhill
            for i in range(dim):
                step[i] = min(0.5 * spans[i], step[i] * 1.15)
            # re-seed simplex around new best and go back to NM
            simplex = [best_x[:]]
            fvals = [best]
            scale = 0.06
            for i in range(dim):
                if time_left() <= 0:
                    return best
                x = best_x[:]
                x[i] += scale * spans[i]
                clip_inplace(x)
                simplex.append(x)
                fvals.append(eval_f(x))
            nm_no_improve = 0
            last_best_nm = best
        else:
            pat_no_improve += 1
            # shrink steps
            for i in range(dim):
                step[i] = max(min_step[i], step[i] * 0.6)

            # If really stuck: heavy-tailed perturbation + one Halton injection
            if pat_no_improve >= 8:
                pat_no_improve = 0

                if time_left() <= 0:
                    return best

                # heavy-tailed jump around best
                x = best_x[:]
                # jump radius based on span and current step (not too small)
                for i in range(dim):
                    base = max(step[i], 0.03 * spans[i])
                    x[i] += heavy_step(base)
                clip_inplace(x)
                fx = eval_f(x)
                if fx < best:
                    best, best_x = fx, x

                # Halton injection
                if time_left() <= 0:
                    return best
                k = 1 + int((time.time() - t0) * 17)  # time-varying index (cheap)
                xh = halton_point(k)
                fh = eval_f(xh)
                if fh < best:
                    best, best_x = fh, xh

                # reset simplex around best and try NM again
                simplex = [best_x[:]]
                fvals = [best]
                scale = 0.09
                for i in range(dim):
                    if time_left() <= 0:
                        return best
                    xi = best_x[:]
                    xi[i] += scale * spans[i]
                    clip_inplace(xi)
                    simplex.append(xi)
                    fvals.append(eval_f(xi))
                nm_no_improve = 0
                last_best_nm = best

    return best
