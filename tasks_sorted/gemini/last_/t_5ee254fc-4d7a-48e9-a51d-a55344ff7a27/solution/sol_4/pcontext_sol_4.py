#To improve upon the previous L-SHADE result, I propose a **Memetic Algorithm: L-SHADE with Coordinate Descent (Local Search)**.
#
#### Why this is better:
#1.  **Exploitation Gap:** The previous result (23.54) indicates the algorithm found the "basin" of the global minimum but failed to drill down to the precise bottom. Pure Differential Evolution (DE) relies on random mutations, which becomes inefficient when very close to the optimal point.
#2.  **Coordinate Descent (The "Polisher"):** I have added a deterministic Local Search mechanism. When the global search (L-SHADE) identifies a new best solution, the algorithm switches to a "polishing" mode. It iteratively tweaks each dimension of the best vector. This is extremely effective for functions like Rosenbrock or Sphere where high precision is needed.
#3.  **Boundary Reflection:** Instead of simply clipping values to bounds (which piles population on the edges), this implementation uses **Reflection**. If a particle hits a wall, it bounces back. This preserves variance and prevents stagnation at the domain limits.
#
#Here is the fully operational, self-contained Python code:
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Memetic Algorithm: L-SHADE (Global Search) hybridized with 
    Coordinate Descent (Local Search) for fine-tuning.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    def get_time_ratio():
        elapsed = (datetime.now() - start_time).total_seconds()
        return elapsed / max_time

    def check_timeout():
        return (datetime.now() - start_time) >= time_limit

    # --- Preprocessing ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- L-SHADE Parameters ---
    # Population size logic: Linear Reduction
    # Start with a decent size to explore, reduce to concentrate
    r_N_init = 18
    N_init = int(r_N_init * dim)
    N_init = np.clip(N_init, 50, 300) # Cap to ensure speed in high dims
    N_min = 4
    
    # Memory Size
    H = 5
    
    # State storage
    state = {
        'pop': None,
        'fit': None,
        'M_CR': np.full(H, 0.5),
        'M_F': np.full(H, 0.5),
        'k_mem': 0,
        'archive': [],
        'N_current': N_init,
        'best_val': float('inf'),
        'best_vec': None,
        'generation': 0
    }

    # --- Helper: Boundary Reflection ---
    def handle_bounds(candidates):
        # Reflect lower
        under_bounds = candidates < min_b
        if np.any(under_bounds):
            candidates[under_bounds] = 2 * min_b[np.where(under_bounds)[1]] - candidates[under_bounds]
        
        # Reflect upper
        over_bounds = candidates > max_b
        if np.any(over_bounds):
            candidates[over_bounds] = 2 * max_b[np.where(over_bounds)[1]] - candidates[over_bounds]
            
        # Hard Clip (in case reflection went out again)
        return np.clip(candidates, min_b, max_b)

    # --- Initialization ---
    state['pop'] = min_b + np.random.rand(state['N_current'], dim) * diff_b
    state['fit'] = np.full(state['N_current'], float('inf'))
    
    for i in range(state['N_current']):
        if check_timeout(): return state['best_val']
        val = func(state['pop'][i])
        state['fit'][i] = val
        if val < state['best_val']:
            state['best_val'] = val
            state['best_vec'] = state['pop'][i].copy()

    # Sort population
    sort_idx = np.argsort(state['fit'])
    state['pop'] = state['pop'][sort_idx]
    state['fit'] = state['fit'][sort_idx]

    # --- Local Search Function (Coordinate Descent) ---
    def local_search_step(current_best_vec, current_best_val, step_scale):
        """
        Tries to improve the best vector by moving along axes.
        step_scale: multiplier for step size, decays over time.
        """
        improved = False
        vec = current_best_vec.copy()
        val = current_best_val
        
        # Determine search step size based on domain width and time
        # We search a subset of dimensions if dim is high to save time
        dims_to_search = list(range(dim))
        if dim > 50:
            random.shuffle(dims_to_search)
            dims_to_search = dims_to_search[:50]

        for d in dims_to_search:
            if check_timeout(): break
            
            step = diff_b[d] * step_scale * 0.05 # 5% of domain width scaled
            
            # Try positive direction
            origin = vec[d]
            vec[d] = np.clip(origin + step, min_b[d], max_b[d])
            new_val = func(vec)
            
            if new_val < val:
                val = new_val
                improved = True
                # If success, keep vec[d] changed
            else:
                # Try negative direction
                vec[d] = np.clip(origin - step, min_b[d], max_b[d])
                new_val = func(vec)
                if new_val < val:
                    val = new_val
                    improved = True
                else:
                    # Revert
                    vec[d] = origin
        
        return vec, val, improved

    # --- Main Loop ---
    while not check_timeout():
        state['generation'] += 1
        
        # 1. Population Size Reduction (L-SHADE)
        time_ratio = get_time_ratio()
        N_target = int(round((N_min - N_init) * time_ratio + N_init))
        N_target = max(N_min, N_target)
        
        if state['N_current'] > N_target:
            # We assume pop is sorted at end of loop, so just truncate end
            remove_count = state['N_current'] - N_target
            state['N_current'] = N_target
            state['pop'] = state['pop'][:N_target]
            state['fit'] = state['fit'][:N_target]
            
            # Reduce Archive
            if len(state['archive']) > N_target:
                # Random removal
                state['archive'] = random.sample(state['archive'], N_target)

        # 2. Parameter Generation
        N = state['N_current']
        pop = state['pop']
        fit = state['fit']
        
        # Select memory indices
        r_idx = np.random.randint(0, H, N)
        m_cr = state['M_CR'][r_idx]
        m_f = state['M_F'][r_idx]
        
        # Generate CR
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F (Cauchy)
        f = m_f + 0.1 * np.random.standard_cauchy(N)
        # Handle F constraints
        f[f > 1.0] = 1.0
        while np.any(f <= 0):
            neg_idx = f <= 0
            f[neg_idx] = m_f[neg_idx] + 0.1 * np.random.standard_cauchy(np.sum(neg_idx))
            f[f > 1.0] = 1.0
            
        # 3. Mutation: current-to-pbest/1
        # Sort is maintained/done at end of loop, so pop[0] is best
        # p-best selection (top p%)
        p_val = np.random.uniform(2/N, 0.2)
        top_p = int(max(1, p_val * N))
        
        pbest_indices = np.random.randint(0, top_p, N) # Since pop is sorted, these are indices in pop
        x_pbest = pop[pbest_indices]
        
        # r1 and r2 generation
        # r1 != i
        r1_indices = np.random.randint(0, N, N)
        collision = (r1_indices == np.arange(N))
        r1_indices[collision] = (r1_indices[collision] + 1) % N
        x_r1 = pop[r1_indices]
        
        # r2 from Union(Pop, Archive)
        archive_arr = np.array(state['archive']) if len(state['archive']) > 0 else np.empty((0, dim))
        if len(archive_arr) > 0:
            union_pop = np.vstack((pop, archive_arr))
        else:
            union_pop = pop
            
        r2_indices = np.random.randint(0, len(union_pop), N)
        # Rough collision check for speed
        collision_r2 = (r2_indices == np.arange(N)) | (r2_indices == r1_indices)
        if np.any(collision_r2):
            r2_indices[collision_r2] = np.random.randint(0, len(union_pop), np.sum(collision_r2))
        x_r2 = union_pop[r2_indices]
        
        # Compute Mutant
        mutant = pop + f[:, None] * (x_pbest - pop) + f[:, None] * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial) & Bounds
        j_rand = np.random.randint(0, dim, N)
        rand_vals = np.random.rand(N, dim)
        mask = rand_vals < cr[:, None]
        mask[np.arange(N), j_rand] = True
        
        trial = np.where(mask, mutant, pop)
        trial = handle_bounds(trial)
        
        # 5. Selection and Updates
        succ_f = []
        succ_cr = []
        diff_f = []
        
        new_pop = pop.copy()
        new_fit = fit.copy()
        
        did_improve_best = False
        
        for i in range(N):
            if check_timeout(): break
            
            val_trial = func(trial[i])
            
            if val_trial <= fit[i]:
                # Add original to archive
                state['archive'].append(pop[i].copy())
                
                # Store success params
                if val_trial < fit[i]:
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    diff_f.append(fit[i] - val_trial)
                
                new_pop[i] = trial[i]
                new_fit[i] = val_trial
                
                if val_trial < state['best_val']:
                    state['best_val'] = val_trial
                    state['best_vec'] = trial[i].copy()
                    did_improve_best = True

        state['pop'] = new_pop
        state['fit'] = new_fit
        
        # Update Archive Size (must not exceed current Pop size)
        if len(state['archive']) > N:
            random.shuffle(state['archive'])
            state['archive'] = state['archive'][:N]
            
        # Update Memory (Lehmer Mean)
        if len(succ_f) > 0:
            succ_f = np.array(succ_f)
            succ_cr = np.array(succ_cr)
            diff_f = np.array(diff_f)
            
            total_diff = np.sum(diff_f)
            weights = diff_f / total_diff
            
            m_cr_new = np.sum(weights * succ_cr)
            if np.isnan(m_cr_new): m_cr_new = 0.5
            
            num_f = np.sum(weights * (succ_f**2))
            den_f = np.sum(weights * succ_f)
            m_f_new = num_f / den_f if den_f > 0 else 0.5
            
            k = state['k_mem']
            state['M_CR'][k] = 0.5 * state['M_CR'][k] + 0.5 * m_cr_new # Weighted update
            state['M_F'][k] = 0.5 * state['M_F'][k] + 0.5 * m_f_new
            state['k_mem'] = (k + 1) % H
            
        # Sort for next iteration (crucial for p-best selection)
        sort_idx = np.argsort(state['fit'])
        state['pop'] = state['pop'][sort_idx]
        state['fit'] = state['fit'][sort_idx]
        
        # --- Local Search Integration ---
        # If we found a new best, or if we are late in the game (refinement phase),
        # try to polish the best vector.
        # Condition: Time > 50% OR Improvement happened.
        if (did_improve_best or time_ratio > 0.5) and not check_timeout():
            
            # Dynamic Step Scale: reduces as time passes (Annealing)
            # Starts at 0.1, ends near 1e-5
            step_scale = (1.0 - time_ratio)**2 + 1e-6
            
            ls_vec, ls_val, ls_success = local_search_step(state['best_vec'], state['best_val'], step_scale)
            
            if ls_success and ls_val < state['best_val']:
                state['best_val'] = ls_val
                state['best_vec'] = ls_vec
                # Inject back into population (replace worst)
                state['pop'][-1] = ls_vec
                state['fit'][-1] = ls_val
                # Re-sort
                sort_idx = np.argsort(state['fit'])
                state['pop'] = state['pop'][sort_idx]
                state['fit'] = state['fit'][sort_idx]

    return state['best_val']
