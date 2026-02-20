import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time

def build_transition_matrix(N, g, a=0.1, gamma=0.8, r=1.0, h=0.44, D=0.0005):
    dx = 1.0 / N
    dy = 1.0 / N
    num_states = (N + 1) * (N + 2) // 2
    
    state_to_idx = np.full((N + 1, N + 1), -1, dtype=int)
    idx_to_state = []
    
    idx = 0
    for i in range(N + 1):
        for j in range(N + 1 - i):
            state_to_idx[i, j] = idx
            idx_to_state.append((i, j))
            idx += 1
            
    row, col, data = [], [], []
    rates = {}
    
    for k in range(num_states):
        i, j = idx_to_state[k]
        
        X = i * dx
        Y = j * dy
        T = 1.0 - X - Y
        
        den = X + T
        if den < 1e-10:
            den = 1e-10
            
        FX = a * X * Y - g * X / den + gamma * X * T
        FY = r * T * Y - h * Y - a * X * Y
        
        rate_x_plus = max(FX, 0) / dx + D / (dx * dx)
        rate_x_minus = max(-FX, 0) / dx + D / (dx * dx)
        rate_y_plus = max(FY, 0) / dy + D / (dy * dy)
        rate_y_minus = max(-FY, 0) / dy + D / (dy * dy)
        
        rates[k] = (rate_x_plus, rate_x_minus, rate_y_plus, rate_y_minus)
        
        diag_val = 0.0
        
        if i + 1 <= N and j <= N - (i + 1):
            next_k = state_to_idx[i+1, j]
            row.append(k); col.append(next_k); data.append(rate_x_plus)
            diag_val += rate_x_plus
            
        if i - 1 >= 0:
            next_k = state_to_idx[i-1, j]
            row.append(k); col.append(next_k); data.append(rate_x_minus)
            diag_val += rate_x_minus
            
        if j + 1 <= N and i <= N - (j + 1):
            next_k = state_to_idx[i, j+1]
            row.append(k); col.append(next_k); data.append(rate_y_plus)
            diag_val += rate_y_plus
            
        if j - 1 >= 0:
            next_k = state_to_idx[i, j-1]
            row.append(k); col.append(next_k); data.append(rate_y_minus)
            diag_val += rate_y_minus
            
        row.append(k); col.append(k); data.append(-diag_val)
        
    M = sp.coo_matrix((data, (row, col)), shape=(num_states, num_states)).tocsc()
    return M, num_states, state_to_idx, idx_to_state, rates, dx, dy

def get_steady_state(M, num_states):
    MT = M.T.tolil()
    MT[-1, :] = 1.0
    b = np.zeros(num_states)
    b[-1] = 1.0
    p = spla.spsolve(MT.tocsc(), b)
    p[p < 1e-30] = 1e-30
    p /= np.sum(p)
    return p

def compute_flux_and_epr(N, g, D):
    M, num_states, state_to_idx, idx_to_state, rates, dx, dy = build_transition_matrix(N, g, D=D)
    P = get_steady_state(M, num_states)
    
    J_X_grid = np.zeros(num_states)
    J_Y_grid = np.zeros(num_states)
    
    for k in range(num_states):
        i, j = idx_to_state[k]
        r_xp, r_xm, r_yp, r_ym = rates[k]
        
        flux_x = 0.0
        if i + 1 <= N and j <= N - (i + 1):
            next_k = state_to_idx[i+1, j]
            r_xm_next = rates[next_k][1]
            net_J = r_xp * P[k] - r_xm_next * P[next_k]
            flux_x += net_J * dx / 2
            J_X_grid[next_k] += net_J * dx / 2
        J_X_grid[k] += flux_x
        
        flux_y = 0.0
        if j + 1 <= N and i <= N - (j + 1):
            next_k = state_to_idx[i, j+1]
            r_ym_next = rates[next_k][3]
            net_J = r_yp * P[k] - r_ym_next * P[next_k]
            flux_y += net_J * dy / 2
            J_Y_grid[next_k] += net_J * dy / 2
        J_Y_grid[k] += flux_y

    cont_epr = 0.0
    cont_flux = 0.0
    for k in range(num_states):
        jx = J_X_grid[k]
        jy = J_Y_grid[k]
        J_mag = np.sqrt(jx**2 + jy**2)
        area = dx * dy
        if P[k] > 1e-15:
            # Note: in non-equilibrium thermodynamics literature, sometimes EPR calculation uses 
            # different forms of D mapping. Here we use J^2 / (D*P).
            cont_epr += (J_mag**2 / (D * P[k])) * area
        cont_flux += J_mag * area
        
    # As an alternative, we calculate discrete EPR to compare or use.
    # We will return discrete EPR too just in case continuous EPR behaves poorly at small probabilities.
    discrete_epr = 0.0
    for k in range(num_states):
        i, j = idx_to_state[k]
        r_xp, r_xm, r_yp, r_ym = rates[k]
        
        if i + 1 <= N and j <= N - (i + 1):
            next_k = state_to_idx[i+1, j]
            F_fwd = r_xp * P[k]
            F_bwd = rates[next_k][1] * P[next_k]
            if F_fwd > 0 and F_bwd > 0:
                discrete_epr += (F_fwd - F_bwd) * np.log(F_fwd / F_bwd)
            
        if j + 1 <= N and i <= N - (j + 1):
            next_k = state_to_idx[i, j+1]
            F_fwd = r_yp * P[k]
            F_bwd = rates[next_k][3] * P[next_k]
            if F_fwd > 0 and F_bwd > 0:
                discrete_epr += (F_fwd - F_bwd) * np.log(F_fwd / F_bwd)
                
    return cont_epr, cont_flux, discrete_epr

def main():
    D_vals = [1e-5, 1e-4, 5e-4, 1e-3, 1e-2]
    N = 80 
    g_vals = np.linspace(0.1, 0.45, 36)
    
    results = {}
    
    for D in D_vals:
        print(f"Computing for D = {D}")
        eprs = []
        fluxes = []
        deprs = []
        t0 = time.time()
        for g in g_vals:
            epr, flux, depr = compute_flux_and_epr(N, g, D)
            eprs.append(epr)
            fluxes.append(flux)
            deprs.append(depr)
        print(f"Finished D = {D} in {time.time() - t0:.2f}s")
        results[D] = {'epr': eprs, 'flux': fluxes, 'depr': deprs}
        
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for idx, D in enumerate(D_vals):
        eprs = results[D]['epr']
        fluxes = results[D]['flux']
        deprs = results[D]['depr']
        
        ax_epr = axes[0, idx]
        # In case cont_epr is noisy, we plot discrete_epr (depr) instead if it looks better, but let's stick to cont_epr for now.
        # Actually discrete_epr represents the exact microscopic entropy production rate of the Markov process,
        # which represents the macroscopic one well. I'll plot cont_epr.
        ax_epr.plot(g_vals, eprs, 'r-')
        ax_epr.set_title(f'D={D}')
        if idx == 0:
            ax_epr.set_ylabel('EPR')
            
        ax_flux = axes[1, idx]
        ax_flux.plot(g_vals, fluxes, 'b-')
        ax_flux.set_xlabel('g')
        if idx == 0:
            ax_flux.set_ylabel('Avg Flux')
            
    plt.tight_layout()
    plt.savefig('figure_4.png', dpi=300)
    print("Saved figure_4.png")

if __name__ == "__main__":
    main()
