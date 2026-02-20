import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import plotly.graph_objects as go
import time
import argparse

def build_transition_matrix(N, g, a=0.1, gamma=0.8, r=1.0, h=0.44, D=0.0005):
    # Grid size
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
            
    row = []
    col = []
    data = []
    
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
        
        diag_val = 0.0
        
        if i + 1 <= N and j <= N - (i + 1):
            next_k = state_to_idx[i+1, j]
            row.append(k)
            col.append(next_k)
            data.append(rate_x_plus)
            diag_val += rate_x_plus
            
        if i - 1 >= 0:
            next_k = state_to_idx[i-1, j]
            row.append(k)
            col.append(next_k)
            data.append(rate_x_minus)
            diag_val += rate_x_minus
            
        if j + 1 <= N and i <= N - (j + 1):
            next_k = state_to_idx[i, j+1]
            row.append(k)
            col.append(next_k)
            data.append(rate_y_plus)
            diag_val += rate_y_plus
            
        if j - 1 >= 0:
            next_k = state_to_idx[i, j-1]
            row.append(k)
            col.append(next_k)
            data.append(rate_y_minus)
            diag_val += rate_y_minus
            
        row.append(k)
        col.append(k)
        data.append(-diag_val)
        
    M = sp.coo_matrix((data, (row, col)), shape=(num_states, num_states)).tocsc()
    return M, num_states, state_to_idx, idx_to_state

def get_steady_state(M, num_states):
    MT = M.T.tolil()
    MT[-1, :] = 1.0
    
    b = np.zeros(num_states)
    b[-1] = 1.0
    
    p = spla.spsolve(MT.tocsc(), b)
    p[p < 1e-30] = 1e-30
    p /= np.sum(p)
    return p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run a quick test")
    args = parser.parse_args()

    N = 100
    if args.test:
        g_vals = np.linspace(0.0, 0.8, 2)
        print("Running in test mode (2 frames only)...")
    else:
        g_vals = np.linspace(0.0, 0.8, 41)
    
    print("Precomputing landscapes...")
    
    all_U = []

    for idx, g in enumerate(g_vals):
        t0 = time.time()
        M, num_states, _, idx_to_state = build_transition_matrix(N, g)
        p = get_steady_state(M, num_states)
        
        U = -np.log(p)
        U -= np.min(U)
        
        U_grid = np.full((N + 1, N + 1), np.nan)
        for k in range(num_states):
            i, j = idx_to_state[k]
            val = U[k]
            if val > 15.0:
                val = 15.0
            U_grid[j, i] = val
            
        all_U.append(U_grid)
        print(f"Computed g={g:.2f} in {time.time()-t0:.2f}s")
        
    x_val = np.linspace(0, 1, N + 1)
    y_val = np.linspace(0, 1, N + 1)

    fig = go.Figure()
    
    fig.add_trace(go.Surface(z=all_U[0], x=x_val, y=y_val, colorscale='Viridis', cmin=0, cmax=15))
    
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Parrotfish grazing rate g: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }
    
    for i, g in enumerate(g_vals):
        step = {
            "args": [
                {"z": [all_U[i]]},
                {"title": f"Population-potential Landscape U for g={g:.2f}"},
                [0] # The trace indices to be updated
            ],
            "label": f"{g:.2f}",
            "method": "update"
        }
        sliders_dict["steps"].append(step)
        
    fig.update_layout(
        title="Population-potential Landscape U for g=" + str(f"{g_vals[0]:.2f}"),
        sliders=[sliders_dict],
        scene=dict(
            xaxis_title='Macroalgae Cover (X)',
            yaxis_title='Coral Cover (Y)',
            zaxis_title='Potential (U)',
            zaxis=dict(range=[0, 15])
        )
    )
    
    fig.write_html("interactive_figure_2.html")
    print("Interactive plot saved to interactive_figure_2.html")

if __name__ == '__main__':
    main()
