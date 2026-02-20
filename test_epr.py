import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


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
    
    epr = 0.0
    flux_av = 0.0
    
    # Calculate JX and JY at each node, or use discrete edge fluxes
    # Let's use continuous approximation: J = F*P - D*nabla P
    # J_x(i,j) = (flux_{i,j -> i+1,j} + flux_{i-1,j -> i,j}) / 2 * dx ?
    # A robust way is discrete flux on edges: 
    # J_{i, i+1} = M_{i+1 <- i} P_i - M_{i <- i+1} P_{i+1}
    # Then J_X = J_{i, i+1} * dx
    
    J_X_grid = np.zeros(num_states)
    J_Y_grid = np.zeros(num_states)
    
    for k in range(num_states):
        i, j = idx_to_state[k]
        r_xp, r_xm, r_yp, r_ym = rates[k]
        
        flux_x = 0.0
        if i + 1 <= N and j <= N - (i + 1):
            next_k = state_to_idx[i+1, j]
            # NET flux from k to next_k
            r_xm_next = rates[next_k][1] # rate_x_minus of next_k
            net_J = r_xp * P[k] - r_xm_next * P[next_k]
            # The flux density J_X is net probability flow per unit length.
            # net_J is probability per unit time. 
            # In the master equation, dP/dt = ...  diff corresponds to -d/dx J_X, so J_X = net_J * dx
            flux_x += net_J * dx / 2 # Distribute to both nodes
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

    for k in range(num_states):
        jx = J_X_grid[k]
        jy = J_Y_grid[k]
        J_sq = jx**2 + jy**2
        
        # Integration element is area associated with node: dx*dy for interior, less for boundaries
        # Just use average dx*dy
        area = dx * dy
        
        #epr += (J_sq / (D * P[k])) * area
        # Continuous formula might be unstable if P is very small.
        # Alternatively, use discrete EPR formula:
        pass
        

    # Discrete EPR formula: 
    discrete_epr = 0.0
    discrete_flux = 0.0
    for k in range(num_states):
        i, j = idx_to_state[k]
        r_xp, r_xm, r_yp, r_ym = rates[k]
        
        if i + 1 <= N and j <= N - (i + 1):
            next_k = state_to_idx[i+1, j]
            F_fwd = r_xp * P[k]
            F_bwd = rates[next_k][1] * P[next_k]
            if F_fwd > 0 and F_bwd > 0:
                discrete_epr += (F_fwd - F_bwd) * np.log(F_fwd / F_bwd)
            
            # The physical flux J_X = (F_fwd - F_bwd) * dx
            discrete_flux += abs((F_fwd - F_bwd) * dx) * (dx * dy) 
            # wait, J = P * V. F_fwd is probability rate. J_x = net_rate * dx.
            
        if j + 1 <= N and i <= N - (j + 1):
            next_k = state_to_idx[i, j+1]
            F_fwd = r_yp * P[k]
            F_bwd = rates[next_k][3] * P[next_k]
            if F_fwd > 0 and F_bwd > 0:
                discrete_epr += (F_fwd - F_bwd) * np.log(F_fwd / F_bwd)
            
            discrete_flux += abs((F_fwd - F_bwd) * dy) * (dx * dy)
            
    # Continuous EPR integration using discrete terms:
    cont_epr = 0.0
    cont_flux = 0.0
    for k in range(num_states):
        jx = J_X_grid[k]
        jy = J_Y_grid[k]
        J_mag = np.sqrt(jx**2 + jy**2)
        area = dx * dy
        if P[k] > 1e-15:
            cont_epr += (J_mag**2 / (D * P[k])) * area
        cont_flux += J_mag * area
        
    return cont_epr, cont_flux, discrete_epr

def main():
    gs = [0.1, 0.2, 0.3, 0.4]
    for g in gs:
        e, f, de = compute_flux_and_epr(50, g, 1e-4) # N=50, D=1e-4
        print(f"g={g:.2f}, EPR={e:.4e}, Flux={f:.4e}, depr={de:.4e}")

if __name__ == "__main__":
    main()
