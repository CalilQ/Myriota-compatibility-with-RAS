import numpy as np
from astropy import units as u


def sample_on_cell_grid_m1583(niters, step_size=3 * u.deg, lat_range=(0 * u.deg, 90 * u.deg)):
    def sample(niters, low_lon, high_lon, low_lat, high_lat):
        """
        This function randomly samples points from a sky cell.

        :param niters: Number of points to sample
        :param low_lon: Lower longitude of sky cell corner
        :param high_lon: Higher longitude of sky cell corner
        :param low_lat: Lower latitude of sky cell corner
        :param high_lat: Higher latitude of sky cell corner
        :return: az, el: Azimuth and elevation of sampled points
        """
        z_low, z_high = np.cos(np.radians(90 - low_lat)), np.cos(np.radians(90 - high_lat))
        az = np.random.uniform(low_lon, high_lon, size=niters)
        el = 90 - np.degrees(np.arccos(
            np.random.uniform(z_low, z_high, size=niters)
        ))
        return az, el

    cell_edges, cell_mids, solid_angles, tel_az, tel_el = [], [], [], [], []

    lat_range = (lat_range[0].to_value(u.deg), lat_range[1].to_value(u.deg))
    ncells_lat = int(
        (lat_range[1] - lat_range[0]) / step_size.to_value(u.deg) + 0.5
    )
    edge_lats = np.linspace(
        lat_range[0], lat_range[1], ncells_lat + 1, endpoint=True
    )
    mid_lats = 0.5 * (edge_lats[1:] + edge_lats[:-1])

    for low_lat, mid_lat, high_lat in zip(edge_lats[:-1], mid_lats, edge_lats[1:]):

        ncells_lon = int(360 * np.cos(np.radians(mid_lat)) / step_size.to_value(u.deg) + 0.5)
        edge_lons = np.linspace(0, 360, ncells_lon + 1, endpoint=True)
        mid_lons = 0.5 * (edge_lons[1:] + edge_lons[:-1])

        solid_angle = (edge_lons[1] - edge_lons[0]) * np.degrees(
            np.sin(np.radians(high_lat)) - np.sin(np.radians(low_lat))
        )
        for low_lon, mid_lon, high_lon in zip(edge_lons[:-1], mid_lons, edge_lons[1:]):
            cell_edges.append((low_lon, high_lon, low_lat, high_lat))
            cell_mids.append((mid_lon, mid_lat))
            solid_angles.append(solid_angle)
            cell_tel_az, cell_tel_el = sample(niters, low_lon, high_lon, low_lat, high_lat)
            tel_az.append(cell_tel_az)
            tel_el.append(cell_tel_el)

    tel_az = np.array(tel_az).T  # TODO, return u.deg
    tel_el = np.array(tel_el).T

    grid_info = np.column_stack([cell_mids, cell_edges, solid_angles])
    grid_info.dtype = np.dtype([  # TODO, return a QTable
        ('cell_lon', np.float), ('cell_lat', np.float),
        ('cell_lon_low', np.float), ('cell_lon_high', np.float),
        ('cell_lat_low', np.float), ('cell_lat_high', np.float),
        ('solid_angle', np.float),
    ])

    return tel_az, tel_el, grid_info[:, 0]