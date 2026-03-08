import matplotlib.pyplot as plt 
import numpy as np 


##################### RNN PERF PLOTTERS #######################################
import numpy as np
import matplotlib.pyplot as plt
z = np.load("A1_mem_rnn_tanh_noclip_final_state.npz")
grad_time = z["grad_time"] # (num_checkpoints, Tstore), NaN padded
sat_time = z["sat_time"] # (num_checkpoints, Tstore)
valid_err = z["valid_error"] # (num_checkpoints,)
rho = z["rho_Whh"] # (num_checkpoints,)
# choose a checkpoint (e.g., last non-empty)
g = grad_time[-1]; s = sat_time[-1]
g = g[np.isfinite(g)]; s = s[np.isfinite(s)]
plt.figure(); plt.hist(np.log10(g + 1e-12), bins=60); plt.title("log10_||dL/dh_t||")
plt.figure(); plt.hist(s, bins=60, range=(0,1)); plt.title("hidden_saturation_distance")
plt.figure(); plt.plot(valid_err); plt.title("validation_error_(%)")
plt.show()

##################### GRU PERF PLOTTERS #######################################


