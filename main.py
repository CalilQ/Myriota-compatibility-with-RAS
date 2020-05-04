## Imports

import os
from contextlib import suppress

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pycraf
from pycraf import conversions as cnv
from pycraf import protection, antenna, geometry
from astropy import units as u, constants as const
from astropy.time import Time
# Note: if the progress bar is not correctly rendered, try the
# following commands in the terminal (requires nodejs):
# jupyter nbextension enable --py widgetsnbextension
# jupyter labextension install @jupyter-widgets/jupyterlab-manager
from astropy.utils.console import ProgressBar
from scipy.stats import percentileofscore
import cysgp4

from create_constellation import create_constellation
from sample_on_cell_grid_m1583 import sample_on_cell_grid_m1583
from sat_gain_func import sat_gain_func
from equatorial_to_topo_approx import equatorial_to_topo_approx

pjoin = os.path.join

FIGPATH = 'fig_pycraf'

with suppress(IOError):
    os.makedirs(FIGPATH)

## Execution flow parameters
plot_sky_grid = False
plot_epfd_cdf = True
display_equatorial = False

## Observer:

observer_name = 'geolat_50deg'
observer_tname = 'RAS @ lat = 50 deg'
obs_lon, obs_lat, obs_alt = 15., 50., 0.  # deg, deg, km
eq_lat_range = (-30 * u.deg, 90 * u.deg)

## Define EPFD simulation meta parameters.

grid_size = 1. * u.deg
niters = 200
time_range, time_resol = 2000, 1  # seconds

## Rx (RAS) parameters
d_rx = 100 * u.m
eta_a_rx = 50 * u.percent

p_lim = -199 * cnv.dB_W
pfd_lim = -194 * cnv.dB_W_m2

ras_bandwidth = 2.95 * u.MHz

## Constellation parameters

# Note, the following constellation/satellite parameters are preliminary. CRAF is still in the process to talk to the
# satellite operators to work out the final technical parameters.

designation = 'myriota_150mhz'
constellation_name = 'Myriota (150 MHz)'
# designation = 'myriota_400mhz'
# constellation_name = 'Myriota (400 MHz)'

# ### Myriota ###

# constellation parameters
altitudes = np.array([600.])  # what about orbit degradation???
inclinations = np.array([97.69] * 6 + [54.] * 10)
sats_per_plane = np.array([2] * 6 + [4] * 10)
raans = np.hstack([np.arange(0, 160, 30), np.arange(0, 330, 36)])
eccentricities = np.array([0.])
arg_of_perigees = np.array([0.])

# general parameters
freq = 151 * u.MHz
# freq = 400 * u.MHz

# Tx (Sat) parameters

# the following numbers are for one channel only!!!
p_tx_carrier = 10 * cnv.dB_W  # one channel of width 4 kHz
# alternatively, 20 kHz with 20% duty cycle
carrier_bandwidth = 4. * u.kHz
spectral_rolloff = -100 * cnv.dBc  # please provide evidence!
duty_cycle = 10 * u.percent  # for downlink, maximum number

# Peak power density in RAS band
p_tx_nu_peak = (p_tx_carrier.to(u.W) / carrier_bandwidth * spectral_rolloff.to(cnv.dimless)).to(u.W / u.Hz)
# Average power density in RAS band
p_tx_nu = p_tx_nu_peak * duty_cycle
# Average power in RAS band
p_tx = p_tx_nu.to(u.W / u.Hz) * ras_bandwidth

## Plot satellite antenna gain

# sat_obs_az, sat_obs_el = 0. * u.deg, np.linspace(-75, 75, 721) * u.deg
# G_tx = sat_gain_func(sat_obs_az, sat_obs_el)

# plt.close()
# fig = plt.figure(figsize=(12, 6))
# plt.plot(sat_obs_el.to_value(u.deg), G_tx.to_value(cnv.dBi), 'k-')
# plt.xlabel('Offset [deg]')
# plt.ylabel('Gain [dBi]')
# plt.grid()
# plt.show()

fig_basename = '{:s}_{:s}'.format(designation, observer_name)

## Define a custom print function, to store results in text files.

def print_info(*args, **kwargs):
    infofile = open(pjoin(FIGPATH, fig_basename + '_info_and_results.txt'), 'w')
    print(*args, **kwargs)
    print(*args, **kwargs, file=infofile)
    infofile.flush()

## Calculate MCL and print interference levels

print_info('P_tx_nu (into RAS band): {:.2f} {:.1f}'.format(p_tx_nu.to(cnv.dBm_MHz), p_tx_nu.to(cnv.dB_W_Hz)))
print_info('P_tx (into RAS band): {:.2e} {:.1f}'.format(p_tx.to(u.W), p_tx.to(cnv.dB_W)))

MCL = p_tx.to_value(cnv.dB_W) - p_lim.to_value(cnv.dB_W)
FSPL = cnv.free_space_loss(altitudes.min() * u.km, freq).to(cnv.dB)
print_info('MCL: {:.2f} dB'.format(MCL))
print_info('Path loss (free space, lowest altitude): {:.2f} dB'.format(-FSPL.to_value(cnv.dB)))

## Preparing the constellation

# want epoch for the following time
mjd_epoch = 58813.5
pydt = cysgp4.PyDateTime.from_mjd(mjd_epoch)
pydt

sats_tle_tuples = create_constellation(
    mjd_epoch, altitudes, inclinations, sats_per_plane,
    raans, eccentricities, arg_of_perigees,
    )

print_info('total number of satellites', len(sats_tle_tuples))

sats_tles = np.array([
    cysgp4.PyTle(*tle)
    for tle in sats_tle_tuples
    ])

## EPFD

start_mjd = mjd_epoch
# start each simulation iteration at a random time, spread over a day
start_times = start_mjd + np.random.uniform(0, 1, niters)
td = np.arange(0, time_range, time_resol) / 86400.  # 2000 s in steps of 1 s
mjds = start_times[:, np.newaxis] + td[np.newaxis]

print_info('niters: {:d}, time steps: {:d}'.format(niters, len(td)))

ras_observer = cysgp4.PyObserver(obs_lon, obs_lat, obs_alt)

## Simulate satellite positions of full constellation for each iteration run.

result = cysgp4.propagate_many(  # see cysgp4 manual for details
    mjds[:, :, np.newaxis],
    sats_tles[np.newaxis, np.newaxis, :],
    ras_observer,
    do_sat_azel=True,
    )

eci_pos = result['eci_pos']
topo_pos = result['topo']
sat_azel = result['sat_azel']

eci_pos_x, eci_pos_y, eci_pos_z = (eci_pos[..., i] for i in range(3))
topo_pos_az, topo_pos_el, topo_pos_dist, _ = (topo_pos[..., i] for i in range(4))
sat_obs_az, sat_obs_el, sat_obs_dist = (sat_azel[..., i] for i in range(3))

## Tropocentric frame

print_info('-' * 80)
print_info('Topocentric frame')
print_info('-' * 80)

# Produce random RAS telescope pointings (within each grid) for each iteration.
tel_az, tel_el, grid_info = sample_on_cell_grid_m1583(niters, step_size=grid_size)

# EPFD calculation; stores mean received pfd for each iteration and sky cell.
p_rx = np.zeros(tel_az.shape, dtype=np.float64)

# doing the calculation for all iterations and sky cells at once
# is very memory consuming; process data in chunks (of sky cells)
chunk_size = 100
nchunks = tel_az.shape[1] // chunk_size + 1

for niter in ProgressBar(niters, ipython_widget=True):
    # calculating the angular separations for every satellite is extremely
    # slow, need to apply visibility masks from the beginning...
    vis_mask = topo_pos_el[niter] > 0

    FSPL = cnv.free_space_loss(sat_obs_dist[niter, vis_mask, np.newaxis] * u.km, freq).to(cnv.dB)
    G_tx = sat_gain_func(
        sat_obs_az[niter, vis_mask, np.newaxis] * u.deg,
        sat_obs_el[niter, vis_mask, np.newaxis] * u.deg,
    )

    for chunk in range(nchunks):
        cells_sl = slice(
            chunk * chunk_size,
            min((chunk + 1) * chunk_size, tel_az.shape[1])
        )

        ang_sep_topo = geometry.true_angular_distance(
            tel_az[niter, np.newaxis, cells_sl] * u.deg,
            tel_el[niter, np.newaxis, cells_sl] * u.deg,
            topo_pos_az[niter, vis_mask, np.newaxis] * u.deg,
            topo_pos_el[niter, vis_mask, np.newaxis] * u.deg,
        )

        G_rx = antenna.ras_pattern(
            ang_sep_topo, d_rx, const.c / freq, eta_a_rx
        )

        # Calculate average received power over the integration time
        p_rx[niter, cells_sl] = np.sum(
            (p_tx.to(cnv.dB_W) + G_tx + FSPL + G_rx).to_value(u.W),
            axis=0
        ) / mjds.shape[1]

# Analyse/visualize the results.
# calculate pfd from received power (receiver gain was already accounted for!)
pfd = cnv.powerflux_from_prx(p_rx * u.W, freq, 0 * cnv.dBi).to(cnv.dB_W_m2)

pfd_lin = pfd.to_value(u.W / u.m ** 2)
pfd_avg = (np.mean(pfd_lin, axis=0) * u.W / u.m ** 2).to(cnv.dB_W_m2)
pfd_98p = (np.percentile(pfd_lin, 98., axis=0) * u.W / u.m ** 2).to(cnv.dB_W_m2)
pfd_max = (np.max(pfd_lin, axis=0) * u.W / u.m ** 2).to(cnv.dB_W_m2)

_pfd_lim_W_m2 = pfd_lim.to_value(u.W / u.m ** 2)
data_loss_per_cell = np.array([
    100 - percentileofscore(pl, _pfd_lim_W_m2, kind='strict')
    for pl in pfd_lin.T
    ])
data_loss = (
    100 - percentileofscore(pfd_lin.flatten(), _pfd_lim_W_m2, kind='strict')
    ) * u.percent

data_loss_per_iteration = np.array([
    100 - percentileofscore(pl, _pfd_lim_W_m2, kind='strict')
    for pl in pfd_lin
    ])
data_loss_mean = np.mean(data_loss_per_iteration)
data_loss_m1s, data_loss_median, data_loss_p1s = np.percentile(
    data_loss_per_iteration, [15.865, 50., 84.135]
    )

print_info(
    'data loss (total): {0.value:.2f} (+{1:.2f} / -{2:.2f}) {0.unit}'.format(
        data_loss,
        data_loss_p1s - data_loss.to_value(u.percent),
        data_loss.to_value(u.percent) - data_loss_m1s,
    ))

bad_cells = np.count_nonzero(pfd_avg > pfd_lim)
print('number of sky cells above threshold: {:d} ({:.2f} %)'.format(
    bad_cells, 100 * bad_cells / len(grid_info)
    ))

## Plot PFD average / cell and data loss / cell
if plot_sky_grid:
    plt.close()
    fig = plt.figure(figsize=(12, 4))
    val = pfd_avg.to_value(cnv.dB_W_m2)
    vmin, vmax = val.min(), val.max()
    val_norm = (val - vmin) / (vmax - vmin)
    plt.bar(
        grid_info['cell_lon_low'],
        height=grid_info['cell_lat_high'] - grid_info['cell_lat_low'],
        width=grid_info['cell_lon_high'] - grid_info['cell_lon_low'],
        bottom=grid_info['cell_lat_low'],
        color=plt.cm.viridis(val_norm),
        align='edge',
        )
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm)
    cbar.set_label('PFD average / cell [dB(W/m2)]')
    plt.title('EPFD {:s} constellation: total data loss: {:.2f}'.format(
        constellation_name, data_loss
        ))
    plt.xlabel('Azimuth [deg]')
    plt.ylabel('Elevation [deg]')
    plt.xlim((0, 360))
    plt.ylim((0, 90))
    plt.savefig(
        pjoin(FIGPATH, '{:s}_skygrid_avg_pfd_horizontal.png'.format(fig_basename)),
        bbox_inches='tight', dpi=100,
        )
    plt.savefig(
        pjoin(FIGPATH, '{:s}_skygrid_avg_pfd_horizontal.pdf'.format(fig_basename)),
        bbox_inches='tight',
        )
    plt.show()

    plt.close()
    fig = plt.figure(figsize=(12, 4))
    val = data_loss_per_cell
    vmin, vmax = val.min(), val.max()
    val_norm = (val - vmin) / (vmax - vmin)
    plt.bar(
        grid_info['cell_lon_low'],
        height=grid_info['cell_lat_high'] - grid_info['cell_lat_low'],
        width=grid_info['cell_lon_high'] - grid_info['cell_lon_low'],
        bottom=grid_info['cell_lat_low'],
        color=plt.cm.viridis(val_norm),
        align='edge',
        )
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm)
    cbar.set_label('Data loss / cell [%]')
    plt.title('EPFD {:s} constellation: total data loss: {:.2f}'.format(
        constellation_name, data_loss
        ))
    plt.xlabel('Azimuth [deg]')
    plt.ylabel('Elevation [deg]')
    plt.xlim((0, 360))
    plt.ylim((0, 90))
    plt.savefig(
        pjoin(FIGPATH, '{:s}_skygrid_data_loss_horizontal.png'.format(fig_basename)),
        bbox_inches='tight', dpi=100,
        )
    plt.savefig(
        pjoin(FIGPATH, '{:s}_skygrid_data_loss_horizontal.pdf'.format(fig_basename)),
        bbox_inches='tight',
        )
    plt.show()

## Calculate margin at 98% percentile

pfd_dist = (np.sort(pfd_lin) * u.W / u.m ** 2).to(cnv.dB_W_m2)
pfd_dist_all = (np.sort(pfd_lin.flatten()) * u.W / u.m ** 2).to(cnv.dB_W_m2)

# Calculate margin at 98% percentile

pfd98p = np.percentile(pfd_lin, 98., axis=1)
pfd98p_all = np.percentile(pfd_lin, 98.)
pfd98p_mean = np.mean(pfd98p)
pfd98p_sigma = np.std(pfd98p, ddof=1)

print_info(
    '98%: pfd = {0.value:.1f} (+{1:.1f} / -{2:.1f}) {0.unit}'.format(
        (pfd98p_all * u.W / u.m ** 2).to(cnv.dB_W_m2),
        np.abs(((1 + pfd98p_sigma / pfd98p_mean) * cnv.dimless).to_value(cnv.dB)),
        np.abs(((1 - pfd98p_sigma / pfd98p_mean) * cnv.dimless).to_value(cnv.dB)),
    ))

print_info(
    'RAS margin = {:.1f} (+{:.1f} / -{:.1f}) dB'.format(
        pfd_lim.to_value(cnv.dB_W_m2) - (pfd98p_all * u.W / u.m ** 2).to_value(cnv.dB_W_m2),
        np.abs(((1 - pfd98p_sigma / pfd98p_mean) * cnv.dimless).to_value(cnv.dB)),
        np.abs(((1 + pfd98p_sigma / pfd98p_mean) * cnv.dimless).to_value(cnv.dB)),
    ))

## Plot PFD CDF
if plot_epfd_cdf:
    plt.close()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(
        pfd_dist.T, 100 * np.linspace(0, 1, pfd_dist.shape[1], endpoint=True),
        'b-', alpha=0.01,
        )
    plt.plot([], [], 'b-', label='CDF (individual iters)')
    plt.plot(
        pfd_dist_all[::20], 100 * np.linspace(0, 1, pfd_dist_all.size, endpoint=True)[::20],
        'k-', label='CDF (all iters)'
        )
    hline = plt.axhline(98., color='r', alpha=0.5)
    vline = plt.axvline(pfd_lim.to_value(cnv.dB_W_m2), color='r', alpha=0.5)
    plt.grid(color='0.8')
    plt.title('EPFD {:s} constellation: total data loss: {:.2f}'.format(
        constellation_name, data_loss
        ))
    plt.xlabel('PFD [dB(W/m2)]')
    plt.ylabel('Cumulative probability [%]')
    plt.legend(*plt.gca().get_legend_handles_labels())
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.text(
        xmin + 2, 95.,
        r'98%: pfd = ${0.value:.1f}^{{+{1:.1f}}}_{{-{2:.1f}}}$ {0.unit}'.format(
            (pfd98p_all * u.W / u.m ** 2).to(cnv.dB_W_m2),
            np.abs(((1 + pfd98p_sigma / pfd98p_mean) * cnv.dimless).to_value(cnv.dB)),
            np.abs(((1 - pfd98p_sigma / pfd98p_mean) * cnv.dimless).to_value(cnv.dB)),
            ), color='red', ha='left', va='top',
        )
    plt.text(
        xmin + 2, 88.,
        r'$\rightarrow$ RAS margin = ${:.1f}^{{+{:.1f}}}_{{-{:.1f}}}$ dB'.format(
            pfd_lim.to_value(cnv.dB_W_m2) - (pfd98p_all * u.W / u.m ** 2).to_value(cnv.dB_W_m2),
            np.abs(((1 - pfd98p_sigma / pfd98p_mean) * cnv.dimless).to_value(cnv.dB)),
            np.abs(((1 + pfd98p_sigma / pfd98p_mean) * cnv.dimless).to_value(cnv.dB)),
            ), color='red', ha='left', va='top',
        )
    plt.text(
        pfd_lim.to_value(cnv.dB_W_m2) + 0.5, ymin + 20,
        r'RA.769 exceeded @ ${0.value:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ {0.unit}'.format(
            100 * u.percent - data_loss,
            data_loss.to_value(u.percent) - data_loss_m1s,
            data_loss_p1s - data_loss.to_value(u.percent),
            ),
        color='red', ha='left', va='bottom', rotation=90.,
        )
    plt.savefig(
        pjoin(FIGPATH, '{:s}_cumulative_data_loss_horizontal.png'.format(fig_basename)),
        bbox_inches='tight', dpi=100,
        )
    plt.savefig(
        pjoin(FIGPATH, '{:s}_cumulative_data_loss_horizontal.pdf'.format(fig_basename)),
        bbox_inches='tight',
        )
    plt.show()

## Equatorial sky
if display_equatorial:
    print_info('-' * 80)
    print_info('Equatorial frame')
    print_info('-' * 80)

    tel_ra, tel_dec, grid_info = sample_on_cell_grid_m1583(
        niters, step_size=grid_size, lat_range=eq_lat_range
        )

    print_info('Latitude range: {} to {}'.format(*eq_lat_range))

    p_rx = np.zeros(tel_ra.shape, dtype=np.float64)
    chunk_size = 100
    nchunks = tel_ra.shape[1] // chunk_size + 1

    for niter in ProgressBar(niters, ipython_widget=True):
        vis_mask = topo_pos_el[niter] > 0

        FSPL = cnv.free_space_loss(sat_obs_dist[niter, vis_mask, np.newaxis] * u.km, freq).to(cnv.dB)
        G_tx = sat_gain_func(
            sat_obs_az[niter, vis_mask, np.newaxis] * u.deg,
            sat_obs_el[niter, vis_mask, np.newaxis] * u.deg,
        )

        sat_pos_az = topo_pos_az[niter, vis_mask, np.newaxis]
        sat_pos_el = topo_pos_el[niter, vis_mask, np.newaxis]
        obstime = Time(mjds[niter], format='mjd')
        _lst_greenwich_deg = obstime.sidereal_time(
            kind='mean', longitude='greenwich'
        ).deg
        lst_greenwich_deg = np.repeat(
            _lst_greenwich_deg[np.newaxis], topo_pos_az.shape[2], axis=0
        ).T[vis_mask]

        for chunk in range(nchunks):
            cells_sl = slice(
                chunk * chunk_size,
                min((chunk + 1) * chunk_size, tel_ra.shape[1])
            )

            tel_az, tel_el = equatorial_to_topo_approx(
                lst_greenwich_deg[:, np.newaxis],
                obs_lon, obs_lat,
                tel_ra[niter, np.newaxis, cells_sl],
                tel_dec[niter, np.newaxis, cells_sl]
            )

            ang_sep = geometry.true_angular_distance(
                tel_az * u.deg,
                tel_el * u.deg,
                sat_pos_az * u.deg,
                sat_pos_el * u.deg,
            )

            G_rx = antenna.ras_pattern(ang_sep, d_rx, const.c / freq, eta_a_rx)

            _p_rx = (p_tx.to(cnv.dB_W) + G_tx + FSPL + G_rx).to_value(u.W)
            tel_horizon_mask = tel_el < 0.
            # for low declinations, depending on time, telescope would need
            # to point to negative elevation, which is impossible, to target
            # astronomical object; set received power for such cases to zero
            _p_rx[tel_horizon_mask] = 1e-29
            p_rx[niter, cells_sl] = np.sum(_p_rx, axis=0) / mjds.shape[1]

    pfd = cnv.powerflux_from_prx(p_rx * u.W, freq, 0 * cnv.dBi).to(cnv.dB_W_m2)

    pfd_lin = pfd.to_value(u.W / u.m ** 2)
    pfd_avg = (np.mean(pfd_lin, axis=0) * u.W / u.m ** 2).to(cnv.dB_W_m2)
    pfd_98p = (np.percentile(pfd_lin, 98., axis=0) * u.W / u.m ** 2).to(cnv.dB_W_m2)
    pfd_max = (np.max(pfd_lin, axis=0) * u.W / u.m ** 2).to(cnv.dB_W_m2)

    data_loss_per_cell = np.array([
        100 - percentileofscore(pl, pfd_lim.to_value(u.W / u.m ** 2), kind='strict')
        for pl in pfd_lin.T
        ])

    bad_cells = np.count_nonzero(pfd_avg > pfd_lim)
    print_info('number of sky cells above threshold: {:d} ({:.2f} %)'.format(
        bad_cells, 100 * bad_cells / len(grid_info)
        ))

    bad2p_cells = np.count_nonzero(data_loss_per_cell > 2.)
    bad5p_cells = np.count_nonzero(data_loss_per_cell > 5.)
    print_info('number of sky cells with more than 2% data loss: {:d} / {:d})'.format(
        bad2p_cells, len(grid_info)
        ))
    print_info('number of sky cells with more than 5% data loss: {:d} / {:d})'.format(
        bad5p_cells, len(grid_info)
        ))

## Plot sky cells for equatorial frame
if plot_sky_grid:
    plt.close()
    fig = plt.figure(figsize=(12, 4))
    val = pfd_avg.to_value(cnv.dB_W_m2)
    vmin, vmax = val.min(), val.max()
    val_norm = (val - vmin) / (vmax - vmin)
    plt.bar(
        grid_info['cell_lon_low'],
        height=grid_info['cell_lat_high'] - grid_info['cell_lat_low'],
        width=grid_info['cell_lon_high'] - grid_info['cell_lon_low'],
        bottom=grid_info['cell_lat_low'],
        color=plt.cm.viridis(val_norm),
        align='edge',
        )
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm)
    cbar.set_label('PFD average / cell [dB(W/m2)]')
    plt.title('EPFD {:s} constellation: Equatorial frame'.format(
        constellation_name
        ))
    plt.xlabel('Right ascension [deg]')
    plt.ylabel('Declination [deg]')
    plt.xlim((0, 360))
    plt.ylim((eq_lat_range[0].to_value(u.deg), eq_lat_range[1].to_value(u.deg)))
    plt.savefig(
        pjoin(FIGPATH, '{:s}_skygrid_avg_pfd_equatorial.png'.format(fig_basename)),
        bbox_inches='tight', dpi=100,
        )
    plt.savefig(
        pjoin(FIGPATH, '{:s}_skygrid_avg_pfd_equatorial.pdf'.format(fig_basename)),
        bbox_inches='tight',
        )
    plt.show()

    plt.close()
    fig = plt.figure(figsize=(12, 4))
    val = data_loss_per_cell
    vmin, vmax = val.min(), val.max()
    val_norm = (val - vmin) / (vmax - vmin)
    plt.bar(
        grid_info['cell_lon_low'],
        height=grid_info['cell_lat_high'] - grid_info['cell_lat_low'],
        width=grid_info['cell_lon_high'] - grid_info['cell_lon_low'],
        bottom=grid_info['cell_lat_low'],
        color=plt.cm.viridis(val_norm),
        align='edge',
        )
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm)
    cbar.set_label('Data loss / cell [%]')
    plt.title('EPFD {:s} constellation: Equatorial frame'.format(
        constellation_name
        ))
    plt.xlabel('Right ascension [deg]')
    plt.ylabel('Declination [deg]')
    plt.xlim((0, 360))
    plt.ylim((eq_lat_range[0].to_value(u.deg), eq_lat_range[1].to_value(u.deg)))
    plt.savefig(
        pjoin(FIGPATH, '{:s}_skygrid_data_loss_equatorial.png'.format(fig_basename)),
        bbox_inches='tight', dpi=100,
        )
    plt.savefig(
        pjoin(FIGPATH, '{:s}_skygrid_data_loss_equatorial.pdf'.format(fig_basename)),
        bbox_inches='tight',
        )
    plt.show()