import numpy as np
import cysgp4


def create_constellation(mjd_epoch, altitudes, inclinations, sats_per_plane, raans, eccs, arg_pergs):
    my_sat_tles = []
    sat_nr = 80000
    altitudes, inclinations, sats_per_plane, raans, eccs, arg_pergs = np.broadcast_arrays(
        altitudes, inclinations, sats_per_plane, raans, eccs, arg_pergs
    )
    for alt, inc, s, rs, ecc, ap in zip(
            altitudes, inclinations, sats_per_plane, raans, eccs, arg_pergs
    ):
        # distribute sats evenly in each plane
        mas = np.linspace(0.0, 360.0, s, endpoint=False)
        # but have a different starting phase per plane
        mas += np.random.uniform(0, 360, 1)
        # mas %= 360.

        mas, rs = np.meshgrid(mas, rs)
        mas, rs = mas.flatten(), rs.flatten()

        mm = cysgp4.satellite_mean_motion(alt)
        for ma, raan in zip(mas, rs):
            my_sat_tles.append(
                cysgp4.tle_linestrings_from_orbital_parameters(
                    'TEST {:d}'.format(sat_nr), sat_nr, mjd_epoch,
                    inc, raan, ecc, ap, ma, mm
                ))

            sat_nr += 1

    return my_sat_tles