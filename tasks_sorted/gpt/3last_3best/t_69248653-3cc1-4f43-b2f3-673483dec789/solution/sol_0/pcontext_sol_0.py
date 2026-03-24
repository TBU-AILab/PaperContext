import random
import math
import time

def run(func, dim, bounds, max_time):
    """
    Time-bounded minimization using a self-contained Adaptive Simulated Annealing
    with occasional random restarts.

    Parameters
    ----------
    func : callable
        func(x) -> float, where x is a list/array of length dim
    dim : int
    bounds : list of (low, high)
    max_time : int/float seconds

    Returns
    -------
    best : float
        Best (minimum) objective value found within max_time.
    """

    # --- helpers ---
    def clip(x):
        for i in range(dim):
            lo, hi = bounds[i]
            if x[i] < lo: x[i] = lo
            elif x[i] > hi: x[i] = hi
        return x

    def rand_point():
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

    def span(i):
        return bounds[i][1] - bounds[i][0]

    # --- init ---
    t0 = time.time()
    deadline = t0 + float(max_time)

    # temperature and step-size settings
    # (kept robust across different objective scales)
    T = 1.0
    Tmin = 1e-12
    alpha = 0.98  # cooling rate (per "epoch")
    epoch_len = max(20, 10 * dim)

    # per-dimension step scales (adapted online)
    step = [0.15 * span(i) if span(i) > 0 else 1.0 for i in range(dim)]
    min_step = [1e-12 * (span(i) if span(i) > 0 else 1.0) for i in range(dim)]
    max_step = [0.5 * (span(i) if span(i) > 0 else 1.0) for i in range(dim)]

    x = rand_point()
    fx = func(x)

    best = fx
    best_x = list(x)

    # stats for adapting step sizes
    accepted = 0
    tried = 0

    # restart management
    no_improve = 0
    restart_patience = max(200, 50 * dim)  # iterations without improvement
    hard_restart_prob = 0.02

    it = 0
    while time.time() < deadline:
        it += 1

        # propose a neighbor (Gaussian step, per-dimension scale)
        xn = list(x)
        for i in range(dim):
            # heavy-tail-ish: mixture of gauss and occasional wider jump
            if random.random() < 0.9:
                delta = random.gauss(0.0, step[i])
            else:
                delta = random.gauss(0.0, 3.0 * step[i])
            xn[i] += delta
        clip(xn)

        fn = func(xn)
        tried += 1

        # SA acceptance
        d = fn - fx
        if d <= 0:
            accept = True
        else:
            # guard against underflow; if T is tiny, acceptance will be ~0
            if T <= Tmin:
                accept = False
            else:
                # exp(-d/T)
                accept = (random.random() < math.exp(-d / T))

        if accept:
            x, fx = xn, fn
            accepted += 1

        # update global best
        if fx < best:
            best = fx
            best_x = list(x)
            no_improve = 0
        else:
            no_improve += 1

        # occasional random restart to escape deep local minima
        if no_improve >= restart_patience or random.random() < hard_restart_prob:
            # restart near the best found (local refine) or fully random
            if random.random() < 0.7:
                x = list(best_x)
                # small shake around best
                for i in range(dim):
                    x[i] += random.gauss(0.0, 0.5 * step[i])
                clip(x)
            else:
                x = rand_point()
            fx = func(x)
            no_improve = 0
            # slightly reheat temperature on restart
            T = max(T, 0.5)

        # epoch-based adaptation and cooling
        if it % epoch_len == 0:
            acc_rate = accepted / tried if tried else 0.0

            # target acceptance around ~0.25-0.35
            if acc_rate > 0.4:
                # steps too small -> increase
                for i in range(dim):
                    step[i] = min(max_step[i], step[i] * 1.25)
            elif acc_rate < 0.2:
                # steps too large -> decrease
                for i in range(dim):
                    step[i] = max(min_step[i], step[i] * 0.75)

            # cool down
            T = max(Tmin, T * alpha)

            # reset counters
            accepted = 0
            tried = 0

            # If we're stuck very late, do a mild reheat to keep exploring
            if (time.time() > (t0 + 0.7 * max_time)) and no_improve > restart_patience // 2:
                T = min(1.0, T * 1.5)

    # return fitness of the best found solution
    return best
