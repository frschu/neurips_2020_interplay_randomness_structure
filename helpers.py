import torch
import numpy as np
import matplotlib.pyplot as plt
import time

def color_iterator(n_iter, cmap='viridis'):
    """ Continuous line coloring. 

    n_iter: int
        Number of color levels = iterations.
        
    cmap: str
        Color map. Check matplotlib for possible values. 
        default: 'viridis'
        
    Returns: 
    scalarMap:
        Generator for colors. Usage:
        ```
            from helper_funcs import color_iterator
            c_iter = color_iterator(n_iter)
            ax.plot(x, y, c=c_iter(idx))
        ```    
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mplcolors
    import matplotlib.cm as cmx
    cm          = plt.get_cmap(cmap) 
    cNorm       = mplcolors.Normalize(vmin=0, vmax=n_iter-1)
    scalarMap   = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    
    c_iter = lambda idx: scalarMap.to_rgba(idx)
    
    return c_iter

def comp_evs(w_recs, max_dim_single=512, n_chunks=10, verbose=False):
    n_rec_epochs, dim_rec, _ = w_recs.shape
    
    # Eigenvalues
    time0 = time.time()
    if dim_rec <= max_dim_single:
        ev_w = np.linalg.eigvals(w_recs)
        ev_w = np.array([sort_complex(evs)[::-1] for evs in ev_w])
    else:
        ev_w = np.zeros((n_rec_epochs, dim_rec), dtype=complex)
        idx_step = int(np.ceil(n_rec_epochs / n_chunks))
        for i in range(n_chunks):
            if verbose:
                print(i, time.time() - time0)
            idx_low = i * idx_step
            idx_up = (i + 1) * idx_step
            ev_w_i = np.linalg.eigvals(w_recs[idx_low:idx_up])
            ev_w_i = np.array([sort_complex(evs)[::-1] for evs in ev_w_i])
            ev_w[idx_low:idx_up] = ev_w_i
    if verbose:
        print("Calculating EVs took %.1f sec." % (time.time() - time0))
    
    return ev_w