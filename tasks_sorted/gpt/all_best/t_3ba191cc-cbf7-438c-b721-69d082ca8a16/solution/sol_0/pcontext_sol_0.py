import random
import time
import math

def run(func, dim, bounds, max_time):
    """
    Time-bounded derivative-free minimization using Differential Evolution (DE/rand/1/bin)
    with occasional random re-initialization to reduce stagnation.

    Parameters
    ----------
    func : callable
        func(x) -> float, where x is a list/sequence of length dim.
    dim : int
        Dimensionality of the search space.
    bounds : list of (low, high)
        Box constraints for each dimension.
    max_time : int or float
        Time budget in seconds.

    Returns
    -------
    best : float
        Best (minimum) objective value found within time budget.
    """
    t0 = time.time()

    # Helpers
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    spans = [highs[i] - lows[i] for i in range(dim)]

    def clip_to_bounds(x):
        for i in range(dim):
            if x[i] < lows[i]:
                x[i] = lows[i]
            elif x[i] > highs[i]:
                x[i] = highs[i]
        return x

    def rand_vec():
        return [lows[i] + random.random() * spans[i] for i in range(dim)]

    # DE parameters (robust defaults)
    pop_size = max(12, 10 * dim)          # typical DE rule of thumb
    F = 0.7                               # mutation factor
    CR = 0.9                              # crossover rate
    restart_patience = max(30, 10 * dim)  # generations without improvement before partial restart
    partial_restart_frac = 0.25           # fraction of population to reinitialize

    # Initialize population
    pop = [rand_vec() for _ in range(pop_size)]
    fits = [func(ind) for ind in pop]

    best_idx = min(range(pop_size), key=lambda i: fits[i])
    best = fits[best_idx]
    best_vec = pop[best_idx][:]

    no_improve_gens = 0

    # Main loop
    gen = 0
    while True:
        if time.time() - t0 >= max_time:
            return best

        improved_this_gen = False
        gen += 1

        # For each target vector, produce a trial vector
        for i in range(pop_size):
            # time check inside loop for tighter budget adherence
            if time.time() - t0 >= max_time:
                return best

            # Choose a, b, c distinct and not equal to i
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)

            xa, xb, xc = pop[a], pop[b], pop[c]
            target = pop[i]

            # Mutation: v = xa + F*(xb - xc)
            v = [xa[j] + F * (xb[j] - xc[j]) for j in range(dim)]
            v = clip_to_bounds(v)

            # Binomial crossover
            j_rand = random.randrange(dim)
            trial = target[:]
            for j in range(dim):
                if random.random() < CR or j == j_rand:
                    trial[j] = v[j]

            # Evaluate and select
            f_trial = func(trial)
            if f_trial <= fits[i]:
                pop[i] = trial
                fits[i] = f_trial

                if f_trial < best:
                    best = f_trial
                    best_vec = trial[:]
                    improved_this_gen = True

        # Stagnation handling: partial restart (keep elites)
        if improved_this_gen:
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        if no_improve_gens >= restart_patience:
            # Keep a small elite set; reinitialize the rest partially
            elite_count = max(2, pop_size // 10)
            elite_idxs = sorted(range(pop_size), key=lambda k: fits[k])[:elite_count]

            # Determine how many to restart
            k_restart = max(1, int(pop_size * partial_restart_frac))
            # Restart the worst individuals (excluding elites)
            candidates = [k for k in sorted(range(pop_size), key=lambda k: fits[k], reverse=True)
                          if k not in elite_idxs]
            for k in candidates[:k_restart]:
                pop[k] = rand_vec()
                fits[k] = func(pop[k])

            # Refresh global best after restart
            best_idx = min(range(pop_size), key=lambda i: fits[i])
            if fits[best_idx] < best:
                best = fits[best_idx]
                best_vec = pop[best_idx][:]
                improved_this_gen = True

            no_improve_gens = 0

        # Occasional local refinement around best (cheap coordinate jitter)
        # Helps in late-stage convergence without gradients.
        if time.time() - t0 >= max_time:
            return best
        if gen % 15 == 0:
            x = best_vec[:]
            fx = best
            # a few jitter trials scaled to bounds
            for _ in range(8):
                if time.time() - t0 >= max_time:
                    return best
                cand = x[:]
                for j in range(dim):
                    # small gaussian-like perturbation using sum of uniforms
                    # scale shrinks over time
                    tau = 0.1 * (1.0 - min(0.999, (time.time() - t0) / max(1e-9, max_time)))
                    noise = (random.random() + random.random() + random.random() - 1.5)  # ~[-1.5,1.5]
                    cand[j] += noise * tau * spans[j]
                cand = clip_to_bounds(cand)
                fc = func(cand)
                if fc < fx:
                    x, fx = cand, fc
                    if fc < best:
                        best, best_vec = fc, cand[:]
#
