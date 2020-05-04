import numpy as np


def equatorial_to_topo_approx(lst_greenwich_deg, obs_lon, obs_lat, ra, dec):
    hour_angle = lst_greenwich_deg - obs_lon - ra

    azim = 180. - np.degrees(np.arctan2(
        np.sin(np.radians(-hour_angle)),
        np.sin(np.radians(obs_lat)) * np.cos(np.radians(hour_angle)) -
        np.cos(np.radians(obs_lat)) * np.tan(np.radians(dec))
    ))
    elev = np.degrees(np.arcsin(
        np.cos(np.radians(obs_lat)) * np.cos(np.radians(hour_angle)) *
        np.cos(np.radians(dec)) +
        np.sin(np.radians(obs_lat)) * np.sin(np.radians(dec))
    ))

    return azim, elev