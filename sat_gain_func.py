import numpy as np
from pycraf import conversions as cnv

def sat_gain_func(sat_obs_az, sat_obs_el):
    # Use 0 dBi antenna for the simulations

    sat_obs_az, sat_obs_el = np.broadcast_arrays(sat_obs_az, sat_obs_el)
    G_tx = np.zeros(sat_obs_az.shape, dtype=np.float64) * cnv.dBi
    return G_tx
