from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_manager import DataManager

import numpy as np
import pandas as pd
import unyt
from sklearn.neighbors import NearestNeighbors
from functools import partial
from astropy import constants as const
from time import perf_counter

from scipy.spatial import KDTree

from octavian.group_properties_calc.group_computations import (
    compute_angular_momentum,
    compute_rotation_quantities,
    compute_radial_quantiles,
    compute_virial_quantities,
)

from octavian.group_properties_calc.group_helpers import (
    sum_per_group,
    count_per_group,
    max_value_per_group,
    max_idx_per_group,
    min_idx_per_group,
    broadcast_to_particles,
    sort_by_group,
    weighted_mean_per_group,
)

# Suppress pandas fragmented frame performance warnings (superfluous)  https://stackoverflow.com/a/76306267
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def _get_group_idx(data: pd.DataFrame, group_data: pd.DataFrame, groupID: str) -> np.ndarray:
  """
  This is more to future-proof things. If data gets reindexed or modified, this should handle things.
  """
  return group_data.index.get_indexer(data[groupID])

def broadcast_properties(data: pd.DataFrame, group_data: pd.DataFrame, groupID: str, properties: list[str] | str, group_idx: np.ndarray = None) -> np.ndarray:
  """
  Broadcasts required properties as a numpy array.
  """
  if group_idx is None:
    group_idx = _get_group_idx(data, group_data, groupID)
  return group_data[properties].to_numpy()[group_idx]

def common_group_properties(data_manager: DataManager, group_name: str, particle_type: str) -> None:
  """
  Computes properties common to all particle types.
  """
  config = data_manager.config
  groupID_key = config['groupIDs'][group_name]
  group_data = data_manager.group_data[group_name]
  n_groups = len(group_data)

  # -
  # step 1: extract arrays from datamanager so the entire operation can be vectorised
  # - 

  if particle_type == 'total':
      ptypes = config['ptypes']
  elif particle_type == 'baryon':
      ptypes = config['ptypes_baryon']
  else:
      ptypes = [particle_type]

  # ids: groups, halos, galaxies
  group_ids_list, halo_ids_list, gal_ids_list = [], [], []
  # physical quantities: mass, potential
  masses_list, potentials_list = [], []
  # kinematics: positions, velocities
  positions_list, velocities_list = [], []

  # cast the required data to numpy.
  for ptype in ptypes:
    df = data_manager.data[ptype]
    group_ids_list.append(df[groupID_key].to_numpy())
    halo_ids_list.append(df['HaloID'].to_numpy())
    gal_ids_list.append(df['GalID'].to_numpy())
    masses_list.append(df['mass'].to_numpy())
    potentials_list.append(df['potential'].to_numpy())
    positions_list.append(df[['x', 'y', 'z']].to_numpy())
    velocities_list.append(df[['vx', 'vy', 'vz']].to_numpy())

  # flat arrays
  group_ids = np.concatenate(group_ids_list)
  halo_ids = np.concatenate(halo_ids_list)
  gal_ids = np.concatenate(gal_ids_list)
  masses = np.concatenate(masses_list)
  potentials = np.concatenate(potentials_list)
  # vstack is equivalent to concatenate here: but these are 3D so this makes the format more explicit
  positions = np.vstack(positions_list)
  velocities = np.vstack(velocities_list)

  # filter particles not assigned to a specific galaxy
  if group_name == 'galaxies':
    keep = gal_ids != -1
    group_ids = group_ids[keep]
    halo_ids = halo_ids[keep]
    masses = masses[keep]
    potentials = potentials[keep]
    positions = positions[keep]
    velocities = velocities[keep]

  # if a group is empty
  if len(masses) == 0:
    return

  # -
  # step 2: group indexing
  # -

  group_idx = group_data.index.get_indexer(group_ids) # build a list of particles where the value of each particle is its group ID

  # parent halo assignment
  if group_name == 'galaxies' and particle_type == 'total':
      parent = np.full(n_groups, -1, dtype=np.int64)
      for i in range(len(group_ids)):
          g = group_idx[i]
          if parent[g] == -1:
              parent[g] = halo_ids[i]
      group_data['parent_halo_index'] = parent

  # -
  # step 3: nparticles
  # - 

  counts = count_per_group(group_idx, n_groups) # number of particles per group
  group_data[f'n{particle_type}'] = counts

  # - 
  # step 4: masses
  # - 

  if particle_type == 'bh':
    group_mass = max_value_per_group(masses, group_idx, n_groups) # REVIEW: why?
  else:
    group_mass = sum_per_group(masses, group_idx, n_groups)

  group_data[f'mass_{particle_type}'] = group_mass

  # -
  # step 5: minimum potential for halos
  # - 

  if group_name == 'halos' and particle_type == 'total':

    minimum_potential_idx = min_idx_per_group(potentials, group_idx, n_groups)
    valid = minimum_potential_idx >= 0 # mask all else away

    minimum_potential_position = np.full((n_groups, 3), np.nan)
    minimum_potential_velocity = np.full((n_groups, 3), np.nan)
    minimum_potential_position[valid] = positions[minimum_potential_idx[valid]]
    minimum_potential_velocity[valid] = velocities[minimum_potential_idx[valid]]

    for i, d in enumerate(['x', 'y', 'z']):
      group_data[f'minpot_{d}'] = minimum_potential_position[:, i]
      group_data[f'minpot_v{d}'] = minimum_potential_velocity[:, i]

  # - 
  # stage 6: centre-of-mass 
  # - 

  com_positions = np.zeros((n_groups, 3))
  com_velocities = np.zeros((n_groups, 3))

  for d in range(3):
    com_positions[:, d] = sum_per_group(positions[:, d] * masses, group_idx, n_groups) / group_mass
    com_velocities[:, d] = sum_per_group(velocities[:, d] * masses, group_idx, n_groups) / group_mass

  for i, d in enumerate(['x', 'y', 'z']):
    group_data[f'{d}_{particle_type}'] = com_positions[:, i]
  for i, d in enumerate(['x', 'y', 'z']):
    group_data[f'v{d}_{particle_type}'] = com_velocities[:, i]

  # -
  # stage 7: relative quantities
  # -

  # halos: potential well, galaxies: centre-of-mass
  if group_name == 'halos':
    if particle_type == 'total':
        ref_positions = minimum_potential_position
        ref_velocities = minimum_potential_velocity
    else:
        ref_positions = group_data[['minpot_x', 'minpot_y', 'minpot_z']].to_numpy()
        ref_velocities = group_data[['minpot_vx', 'minpot_vy', 'minpot_vz']].to_numpy()
  else:
      ref_positions = com_positions
      ref_velocities = com_velocities

  positions_rel = positions - broadcast_to_particles(ref_positions, group_idx)
  velocities_rel_com = velocities - broadcast_to_particles(com_velocities, group_idx)
  velocities_rel_ref = velocities - broadcast_to_particles(ref_velocities, group_idx)

  radii = np.linalg.norm(positions_rel, axis=1)

  # -
  # step 8: velocity dispersion
  # -

  disp_sums = sum_per_group(np.sum(velocities_rel_com**2, axis=1), group_idx, n_groups)
  vel_disps = np.sqrt(disp_sums / counts)
  group_data[f'velocity_dispersion_{particle_type}'] = vel_disps

  # -
  # step 9: angular momentum and rotation
  # -

  L, ktot = compute_angular_momentum(positions_rel, velocities_rel_ref, masses, group_idx, n_groups)
  
  for i, d in enumerate(['x', 'y', 'z']):
    group_data[f'L{d}_{particle_type}'] = L[:, i]

  L_mag = np.linalg.norm(L, axis=1)
  group_data[f'L_{particle_type}'] = L_mag
  group_data[f'ALPHA_{particle_type}'] = np.arctan2(L[:, 1], L[:, 2])
  group_data[f'BETA_{particle_type}'] = np.arcsin(L[:, 0] / L_mag)

  counter_mass, krot, ktot = compute_rotation_quantities(
      positions_rel, velocities_rel_ref, masses, group_idx, L, n_groups
  )

  group_data[f'BoverT_{particle_type}'] = 2 * counter_mass / group_mass
  group_data[f'kappa_rot_{particle_type}'] = krot / ktot

  angular_cols = [
    f'velocity_dispersion_{particle_type}',
    f'Lx_{particle_type}', f'Ly_{particle_type}', f'Lz_{particle_type}',
    f'BoverT_{particle_type}', f'kappa_rot_{particle_type}',
    ]
  
  # for small groups: set quantites = 0 as they are not meaningful (as done in original code)
  small = counts < 3
  for col in angular_cols:
      vals = group_data[col].to_numpy().copy()
      vals[small] = 0.
      group_data[col] = vals

  # -
  # step 10: radial quantities
  # -

  order, unique_ids, g_start, g_end = sort_by_group(group_idx)
  radii_sorted = radii[order]
  masses_sorted = masses[order]

  # from previous code
  quantiles = np.array([0.2, 0.5, 0.8])
  quantile_names = ['r20', 'half_mass', 'r80']

  radial_results = compute_radial_quantiles(radii, masses, group_idx, n_groups, quantiles)

  for q, col_name in enumerate(quantile_names):
      group_data[f'radius_{particle_type}_{col_name}'] = radial_results[:, q]

  # - 
  # step 11: virial quantities for halos
  # - 

  if group_name == 'halos' and particle_type == 'total':
    # from previous code
    group_data['r200'] = data_manager.simulation['r200_factor'] * group_mass**(1/3)
    G_factor = unyt.G.to('(km**2 * kpc)/(Msun * s**2)').d
    group_data['circular_velocity'] = np.sqrt(G_factor * group_mass / group_data['r200'])
    group_data['temperature'] = 3.6e5 * (group_data['circular_velocity'] / 100.0)**2
    group_data['spin_param'] = L_mag / (
        np.sqrt(2) * group_mass * group_data['circular_velocity'].to_numpy() * group_data['r200'].to_numpy()
    )

    rhocrit = (
        data_manager.simulation['rhocrit'] *
        data_manager.create_unit_quantity('rhocrit')
    ).to('Msun / (kpc*a)**3').d

    factors = np.array([200., 500., 2500.])

    virial_r, virial_m = compute_virial_quantities(radii, masses, group_idx, n_groups, rhocrit, factors)

    for f, factor in enumerate([200, 500, 2500]):
        group_data[f'radius_{factor}_c'] = virial_r[:, f]
        group_data[f'mass_{factor}_c'] = virial_m[:, f]

def gas_group_properties(data_manager: DataManager, group_name: str) -> None:
    
  config = data_manager.config
  group_data = data_manager.group_data[group_name]
  groupID_key = config['groupIDs'][group_name]

  # load relevant quantities
  df = data_manager.data['gas']
  group_ids = df[groupID_key].to_numpy()
  masses = df['mass'].to_numpy()
  metallicities = df['metallicity'].to_numpy()
  sfrs = df['sfr'].to_numpy()
  temperatures = df['temperature'].to_numpy()
  rhos = df['rho'].to_numpy()
  mass_HI = df['mass_HI'].to_numpy()
  mass_H2 = df['mass_H2'].to_numpy()

  if group_name == 'galaxies':
    keep = df['GalID'].to_numpy() != -1
    group_ids = group_ids[keep]
    masses = masses[keep]
    metallicities = metallicities[keep]
    sfrs = sfrs[keep]
    temperatures = temperatures[keep]
    rhos = rhos[keep]
    mass_HI = mass_HI[keep]
    mass_H2 = mass_H2[keep]

  # guard against empty group
  if len(masses) == 0:
    return
  
  n_groups = len(group_data)
  group_idx = group_data.index.get_indexer(group_ids)

  # HI, H2 masses
  group_data['mass_HI'] = sum_per_group(mass_HI, group_idx, n_groups)
  group_data['mass_H2'] = sum_per_group(mass_H2, group_idx, n_groups)

  # SFR
  group_data['sfr'] = sum_per_group(sfrs, group_idx, n_groups)

  # metallicity
  group_mass = group_data[f'mass_gas'].to_numpy()
  group_sfr = group_data['sfr'].to_numpy()

  group_data['metallicity_mass_weighted'] = sum_per_group(metallicities * masses, group_idx, n_groups) / group_mass
  group_data['metallicity_sfr_weighted'] = sum_per_group(metallicities * sfrs, group_idx, n_groups) / group_sfr

  # CGM quantities
  cgm_mask = rhos < config['nHlim']
  cgm_idx = group_idx[cgm_mask]
  cgm_masses = masses[cgm_mask]
  cgm_temps = temperatures[cgm_mask]
  cgm_metals = metallicities[cgm_mask]

  group_data['mass_cgm'] = sum_per_group(cgm_masses, cgm_idx, n_groups)
  cgm_mass = group_data['mass_cgm'].to_numpy()

  # temperatures
  temp_mass_weighted = sum_per_group(temperatures * masses, group_idx, n_groups)
  group_data['temp_mass_weighted'] = temp_mass_weighted / group_mass

  cgm_temp_mass = sum_per_group(cgm_temps * cgm_masses, cgm_idx, n_groups)
  cgm_temp_metal = sum_per_group(cgm_temps * cgm_masses * cgm_metals, cgm_idx, n_groups)

  group_data['temp_mass_weighted_cgm'] = cgm_temp_mass / cgm_mass
  group_data['temp_metal_weighted_cgm'] = cgm_temp_metal / cgm_temp_mass

  # CGM metallicity
  cgm_metal_mass = sum_per_group(cgm_metals * cgm_masses, cgm_idx, n_groups)
  group_data['metallicity_mass_weighted_cgm'] = cgm_metal_mass / cgm_mass
  group_data['metallicity_temp_weighted_cgm'] = cgm_temp_metal / cgm_metal_mass

def star_group_properties(data_manager: DataManager, group_name: str) -> None:
  config = data_manager.config
  group_data = data_manager.group_data[group_name]
  groupID_key = config['groupIDs'][group_name]

  df = data_manager.data['star']
  group_ids = df[groupID_key].to_numpy()
  masses = df['mass'].to_numpy()
  metallicities = df['metallicity'].to_numpy()
  ages = df['age'].to_numpy()

  if group_name == 'galaxies':
      keep = df['GalID'].to_numpy() != -1
      group_ids = group_ids[keep]
      masses = masses[keep]
      metallicities = metallicities[keep]
      ages = ages[keep]

  if len(masses) == 0:
      return

  n_groups = len(group_data)
  group_idx = group_data.index.get_indexer(group_ids)

  # metallicity
  metal_mass = sum_per_group(metallicities * masses, group_idx, n_groups)
  total_mass = sum_per_group(masses, group_idx, n_groups)
  group_data['metallicity_stellar'] = metal_mass / total_mass

  # age
  group_mass_star = group_data['mass_star'].to_numpy()
  group_data['age_mass_weighted'] = sum_per_group(ages * masses, group_idx, n_groups) / group_mass_star
  group_data['age_metal_weighted'] = sum_per_group(ages * masses * metallicities, group_idx, n_groups) / metal_mass

def bh_group_properties(data_manager: DataManager, group_name: str) -> None:
  config = data_manager.config
  group_data = data_manager.group_data[group_name]
  groupID_key = config['groupIDs'][group_name]

  df = data_manager.data['bh']
  group_ids = df[groupID_key].to_numpy()
  masses = df['mass'].to_numpy()
  bhmdots = df['bhmdot'].to_numpy()

  if group_name == 'galaxies':
      keep = df['GalID'].to_numpy() != -1
      group_ids = group_ids[keep]
      masses = masses[keep]
      bhmdots = bhmdots[keep]

  if len(masses) == 0:
      return

  n_groups = len(group_data)
  group_idx = group_data.index.get_indexer(group_ids)

  # find most massive BH per group
  max_idx = max_idx_per_group(masses, group_idx, n_groups)
  valid = max_idx >= 0

  # bhmdot of most massive BH
  bhmdot = np.full(n_groups, np.nan)
  bhmdot[valid] = bhmdots[max_idx[valid]]
  group_data['bhmdot'] = bhmdot

  # Eddington fraction
  FRAD = 0.1
  edd_factor = (4 * np.pi * const.G * const.m_p / (FRAD * const.c * const.sigma_T)).to('1/yr').value

  bh_mass = np.full(n_groups, np.nan)
  bh_mass[valid] = masses[max_idx[valid]]
  group_data['bh_fedd'] = bhmdot / (edd_factor * bh_mass)

def calculate_local_densities(data_manager: DataManager) -> None:

  config = data_manager.config
  groups = config['groups']

  for group in groups:
    group_data = data_manager.group_data[group]

    if len(group_data) == 0:
      print(f"No group data!")
      continue

    pos = group_data[['x_total', 'y_total', 'z_total']].to_numpy()
    mass = group_data['mass_total'].to_numpy()

    # REVIEW: moving a FOF6D optimisation into this function
    # previously the pandas .explode() calls were memory-intensive
    tree = KDTree(pos)
    for radius in [300., 1000., 3000.]:
      volume = 4./3. * np.pi * radius**3
      index_lists = tree.query_ball_point(pos, radius, workers=-1) # workers=-1 means all processors are used (from documentation)
      mass_sums = np.array([mass[il].sum() for il in index_lists])
      counts = np.array([len(il) for il in index_lists])

      group_data[f'local_mass_density_{int(radius)}'] = mass_sums / volume
      group_data[f'local_number_density_{int(radius)}'] = counts / volume


def calculate_group_properties(data_manager: DataManager) -> None:
  config = data_manager.config
  for ptype in config['ptypes']:
    data_manager.load_property('potential', ptype)

  groups = config['groups']

  columns_to_drop = ['vx', 'vy', 'vz', 'potential']
  to_process = config['to_process']

  # total common
  if 'total' in to_process:
    for group in groups:
      common_group_properties(data_manager, group, 'total')


  # dm common
  if 'dm' in to_process:
    for group in groups:
      common_group_properties(data_manager, group, 'dm')

    data_manager.data['dm'].drop(columns=columns_to_drop, inplace=True)


  # baryon common
  if 'baryon' in to_process:
    for group in groups:
      common_group_properties(data_manager, group, 'baryon')


  # gas common
  if 'gas' in to_process:
    for group in groups:
      common_group_properties(data_manager, group, 'gas')
  
    data_manager.data['gas'].drop(columns=columns_to_drop, inplace=True)


  # star common
  if 'star' in to_process:
    for group in groups:
      common_group_properties(data_manager, group, 'star')

    data_manager.data['star'].drop(columns=columns_to_drop, inplace=True)


  # bh common
  if 'bh' in to_process:
    for group in groups:
      common_group_properties(data_manager, group, 'bh')

    data_manager.data['bh'].drop(columns=columns_to_drop, inplace=True)

  # gas
  if 'gas' in to_process:
    for property in ['rho', 'nh', 'fH2', 'metallicity', 'sfr', 'temperature']:
      data_manager.load_property(property, 'gas')

    # gas massses
    data = data_manager.data['gas']
    data['fHI'] = data.eval('nh / mass')
    not_conserving_mass = data.eval('(fHI + fH2) > 1')
    data.loc[not_conserving_mass, 'fHI'] = 1. - data.loc[not_conserving_mass, 'fH2']

    data['mass_HI'] = config['XH'] * data['fHI'] * data['mass']
    data['mass_H2'] = config['XH'] * data['fH2'] * data['mass']

    for group in groups:
      gas_group_properties(data_manager, group)

  # star
  if 'star' in to_process:
    for property in ['age', 'metallicity']:
      data_manager.load_property(property, 'star')

    for group in groups:
      star_group_properties(data_manager, group)

  # bh
  if 'bh' in to_process:
    for property in ['bhmdot']:
      data_manager.load_property(property, 'bh')
    
    star_props_columns = ['HaloID', 'GalID', 'ptype', 'mass', 'bhmdot']
    for group in groups:
      bh_group_properties(data_manager, group)

  # aperture
  if 'apertures' in to_process and 'galaxies' in config['groups']:
    aperture_props_columns = ['HaloID', 'GalID', 'ptype', 'mass', 'x',  'y', 'z']
    data = pd.concat([data_manager.data[ptype][aperture_props_columns] for ptype in config['ptypes']])

    aperture_HI_columns = ['HaloID', 'GalID', 'ptype', 'mass_HI', 'x',  'y', 'z']
    HI_gas = data_manager.data['gas'][aperture_HI_columns].copy()
    HI_gas.rename(columns={'mass_HI': 'mass'}, inplace=True)
    HI_gas['ptype'] = 'HI'

    aperture_H2_columns = ['HaloID', 'GalID', 'ptype', 'mass_H2', 'x',  'y', 'z']
    H2_gas = data_manager.data['gas'][aperture_H2_columns].copy()
    H2_gas.rename(columns={'mass_H2': 'mass'}, inplace=True)
    H2_gas['ptype'] = 'H2'

    data = pd.concat([data, HI_gas, H2_gas], ignore_index=True)

    aperture = 30.
    galaxy_positions = data_manager.group_data['galaxies'][['x_total', 'y_total', 'z_total']].to_numpy()

    process_halo = partial(calculate_aperture_masses, aperture=aperture, galaxy_positions=galaxy_positions)
    aperture_masses = data.groupby(by='HaloID').apply(process_halo, include_groups = False).reset_index(names=['HaloID', 'GalID'])
    aperture_masses.set_index('GalID', inplace=True)

    for ptype in config['ptypes']:
      data_manager.group_data['galaxies'][f'mass_{ptype}_30kpc'] = aperture_masses[ptype]
    data_manager.group_data['galaxies']['mass_HI_30kpc'] = aperture_masses['HI']
    data_manager.group_data['galaxies']['mass_H2_30kpc'] = aperture_masses['H2']
    data_manager.group_data['galaxies']['mass_total_30kpc'] = data_manager.group_data['galaxies'][[f'mass_{ptype}_30kpc' for ptype in config['ptypes']]].sum(axis=1)

  if 'local_densities' in to_process:
    calculate_local_densities(data_manager)
